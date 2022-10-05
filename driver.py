from Feature_Pipeline.utilities import csv_write_field_header, csv_write_record, select_keys, tamu_select_features
from Feature_Pipeline.grobid_client.grobid_client import run_grobid
from Feature_Pipeline.extractor import TEIExtractor
from Feature_Pipeline.tamu_features.adapter import get_tamu_features
from Feature_Pipeline.p_value import extract_p_values
from Market.set_initial_configuration import InitialConfiguration
from Market.config import agent_weights_file_location, output_folder_location
from Market.Test import test_on_batch, train_ga
from collections import namedtuple
from os import system, name, path, makedirs
import multiprocessing as mp
import statsmodels.api as sm
import pandas as pd
import numpy as np
import shutil
import csv
import yaml


def get_confidence_for_market_score(all_market_scores, median_score, total_agent_participantion):
    """
        :brief Computes the confidence in our confidence scores by using a dimensional interpretation
                i)  Number of participating Agents for a test data point.
                ii) Difference between the final market score with lower (25%) and upper range (75%)
                        * if the difference is lesser than 0.1, only then we get a high-confidence in our confidence
                            scores.
                        * if the difference is lesser than 0.2, then we can get a medium-confidence in our confidence
                            scores.

        :TODO (Research Aspect) In my opinion, this method might have to undergo changes as this might not be an
              accurate way of estimating confidence in our confidence scores. Primarily, while the method relies on the
              consistency between the models, it does not incorporate the variability in the input ground-truth scores.
              The input-label variance might attribute towards output variation as well.

        :param all_market_scores: scores from all the markets
        :param median_score: median market score
        :param total_agent_participation: total number of agents that participated in this market

        :return A string indiciating the Confidence-Level (or confidence in our confidence scores)
    """
    # check for 25% first ; then 75%
    lower_range = np.percentile(all_market_scores, 25)
    upper_range = np.percentile(all_market_scores, 75)

    # criteria for a high confidence - 3 or more agents participating in the market ; range is closer to median score
    if np.abs((lower_range/median_score) - 1) < 0.1 and np.abs((upper_range/median_score) - 1) < 0.1:
        if total_agent_participantion >= 3:
            confidence_result = "High Confidence"
        elif total_agent_participantion > 0:
            confidence_result = "Medium Confidence"
        else:
            confidence_result = "Low Confidence"
    elif np.abs((lower_range/median_score) - 1) < 0.2 and np.abs((upper_range/median_score) - 1) < 0.2:
        if total_agent_participantion >= 3:
            confidence_result = "Medium Confidence"
        else:
            confidence_result = "Low Confidence"
    else:
        confidence_result = "Low Confidence"

    return confidence_result


def limit_cycle_check(price_history, alpha=0.05):
    """
        :brief Checks the variation in market scores against increasing market rounds (time). This is done to determine
               if there is monotonic increase or decrease in the market score OR if the market score has reached a
               plateau. If p-value > 0.05, then limit-cycle is detected

        :param price_history: comprises of market price variation as seen for all market rounds.
        :param alpha: p-value significance level.

        :return True if it market falls in a limit-cycle situation ; Limit Cycle detected
    """
    # select the last 25 elements to decide if it is a limit-cycle
    price_history = dict(list(price_history.items())[-25:])
    X = np.array(list(price_history.keys())).reshape(-1, 1)
    y = np.array(list(price_history.values())).reshape(-1, 1)

    # fit a linear regressor and check the p-value or significance level of the coefficients of the model
    X2 = sm.add_constant(X)
    est = sm.OLS(y, X2)
    est2 = est.fit()
    if len(est2.pvalues) > 1:
        p_value = est2.pvalues[1]
    else:   # in-case of a constant ; code should not ideally come here as we are not handling the case of only 0.5.
        p_value = est2.pvalues[0]

    # if p-value > 0.05 ; then it isn't statistically significant and we couldn't find a correlation between
    # the market-price and market rounds ;
    if p_value > alpha:
        return True, p_value

    return False, p_value


def get_score(pdf_file, meta_file=None):
    """
        :assumption Is that market model is trained and the corresponding agent weight files are already available
        :brief *) Takes in a PDF file
               *) Extracts features from Feature_Pipeline
               *) Gets the final market prediction for this PDF along with all interpretability aspects.
        :param pdf_file: Input PDF file
        :param meta_file: Metadata CSV if available (used if API invoked at the /claim/meta endpoint)

        :return Dictionary comprising of the following details - final market prediction, limit-cycle (yes or no),
                price-history, interpretability (all levels), all_market_scores (if more than one market is trained),
                confidence on our predicted scores, test data point features itself.
    """

    config_file = open(r'app_config.yaml')
    config = yaml.safe_load(config_file)
    # Run through GROBID - Gen XML
    with open('./pdfs/{0}'.format(pdf_file.filename), 'wb') as f:
        shutil.copyfileobj(pdf_file.file, f)
        f.close()

    # Write meta file to server
    if meta_file:
        with open('./meta/{0}'.format(meta_file.filename), 'wb') as f:
            shutil.copyfileobj(meta_file.file, f)
            f.close()

    # DB cache for co-citation features
    # database = Database('./database')

    run_grobid('./pdfs', './tei_xmls', 1)
    # Generate TXT file - PDFTOTEXT
    if name == 'nt':
        windows_pdftotext_path = config['PIPELINE']['PDFTOTEXT_PATH']
        command = r"{0} .\pdfs/{1}".format(windows_pdftotext_path, pdf_file.filename)
    else:
        command = "pdftotext ./pdfs/{0}".format(pdf_file.filename)
    system(command)
    # Feature Extraction -- keep the features fixed and vary the scores .. total of 7 scores
    feature_list = config['PIPELINE']['FEATURES']
    fields = tuple(feature_list)

    # fields = ("ta3_pid", "doi", "title", "num_citations", "author_count", "sjr", "u_rank", "self_citations",
    #                  "upstream_influential_methodology_count", "subject", "subject_code", "citationVelocity",
    #                  "influentialCitationCount", "references_count", "openaccessflag", "normalized_citations",
    #                  "influentialReferencesCount", "reference_background", "reference_result", "reference_methodology",
    #                  "citations_background", "citations_result", "citations_methodology", "citations_next",	"coCite2",
    #                  "coCite3", "reading_score", "subjectivity", "sentiment", "num_hypo_tested", "real_p",
    #                  "real_p_sign", "p_val_range", "num_significant", "sample_size", "extend_p", "funded",
    #                  "avg_pub", "avg_hidx", "avg_auth_cites", "avg_high_inf_cites",
    #                  "sentiment_agg", "age")

    record = namedtuple('record', fields)
    record.__new__.__defaults__ = (None,) * len(fields)
    # CSV output file
    with open('./test_csvs/test.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header = list(fields)
        csv_write_field_header(writer, header)
        print("Processing ", pdf_file)

        extractor = TEIExtractor('tei_xmls/' + pdf_file.filename.replace('pdf', 'tei.xml'), None)
        extraction_stage = extractor.extract_paper_info()

        issn = extraction_stage.pop('ISSN')
        auth = extraction_stage.pop('authors')
        citations = extraction_stage.pop('citations')

        # Extract p-values
        p_val_stage = extract_p_values('pdfs/' + pdf_file.filename.replace('pdf', 'txt'))
        features = dict(**extraction_stage, **p_val_stage)

        # Add paper id if meta file was uploaded
        if meta_file:
            features['ta3_pid'] = pdf_file.filename.split('_')[-1].replace('.pdf', '')
            meta_file_path = './meta/{0}'.format(meta_file.filename)
            tamu_features = get_tamu_features(meta_file_path, features['ta3_pid'], issn, auth, citations, None)
        else:
            features['ta3_pid'] = pdf_file.filename.split('_')[-1].replace('.pdf', '')+'no_meta'
            tamu_features = get_tamu_features(None, None, issn, auth, citations, None)

        select_tamu_features = select_keys(tamu_features, tamu_select_features)
        features.update(select_tamu_features)
        # Remove venue features
        # TODO: This was a temporary hotfix to remove venue features. Remove in the future.
        remove_venue_features = ["Venue_Citation_Count", "Venue_Scholarly_Output", "Venue_Percent_Cited",
                                 "Venue_CiteScore", "Venue_SNIP", "Venue_SJR"]

        [features.pop(key) for key in remove_venue_features]

        try:
            csv_write_record(writer, features, header)
            f.close()
        except UnicodeDecodeError:
            print("CSV WRITE ERROR")

    # Evaluate claim through the Market
    test_data = pd.read_csv('./test_csvs/test.csv')
    print("Total number of test data points", test_data.shape[0])
    test_feature_vectors = test_data.iloc[:, 3:]
    test_feature_vectors = np.array(test_feature_vectors.values.tolist())

    # assuming - one claim per batch evaluation
    return_test_datapoint = test_data.iloc[:, 3:].to_dict('records')[0]

    all_market_predictions = {}
    model_id = 0
    all_market_scores = []

    # Get input maps for parallel market runs
    input_map = ((test_feature_vectors, agent_weight, True) for agent_weight in agent_weights_file_location)
    # Create thread pool
    t_pool = mp.Pool()
    # To Do: test the added interpret_results param to the input_map so that we ensure front end can use interpretation
    # Evaluate through markets
    market_results = t_pool.starmap(test_on_batch, input_map)
    t_pool.close()
    # Wait for execution across markets to finish
    t_pool.join()

    # Format results
    for market_result in market_results:
        all_market_predictions[model_id] = {}
        all_market_predictions[model_id]["score"] = market_result[0][0]
        # Since one-claim at a time
        all_market_predictions[model_id]["interpret"] = market_result[1][0]
        all_market_predictions[model_id]["price_history"] = market_result[2]
        all_market_scores.append(all_market_predictions[model_id]["score"])
        model_id += 1

    model_id = 0
    # code to get the right model
    all_market_scores = np.array(all_market_scores)
    median_score = np.median(all_market_scores)

    # if median is 0.5 but there is another-value that isn't 0.5 - return that scored market
    if median_score == 0.5 and np.any(all_market_scores != 0.5):
        # get the first market model where the score isn't 0.5
        model_id = np.where(all_market_scores != 0.5)[0][0]
        return_dict = all_market_predictions[model_id]
    # if every_value is 0.5, then return the first market model
    elif median_score == 0.5 and np.all(all_market_scores == 0.5):
        return_dict = all_market_predictions[model_id]
    # if median is not 0.5,then return the median market
    else:
        print(all_market_scores)
        print(np.where(all_market_scores == median_score))
        model_id = np.where(all_market_scores == median_score)[0][0]
        return_dict = all_market_predictions[model_id]

    # check for limit cycles if the final market-score of selected model oscillates between 0.46 to 0.54 and **not 0.5!
    # use price-history relevant to this model
    limit_cycle_flag = False
    p_value = 0
    if return_dict["score"] != 0.5 and (0.46 <= return_dict["score"] <= 0.54):
        limit_cycle_flag, p_value = limit_cycle_check(list(return_dict['price_history'].values())[0])
        if limit_cycle_flag:
            print("Market entered limit cycle!")

    participant_count_string = return_dict["interpret"]["Level Two"]["Total Agents"]
    total_participant_count = [int(s) for s in participant_count_string.split() if s.isdigit()]
    total_participant_count = total_participant_count[0]

    confidence_on_result = get_confidence_for_market_score(all_market_scores, median_score, total_participant_count)
    final_payload = {"score": return_dict["score"], "interpret": return_dict["interpret"],
                     "price_history": return_dict['price_history'],
                     "all market scores": list(all_market_scores), "confidence on market output": confidence_on_result,
                     "test_data_points": return_test_datapoint}
    if limit_cycle_flag:
        final_payload[
            "limit_cycle_detection"] = "Market-price oscillates around this value. For the price-oscillation test, we " \
                                       "obtained a p-value of " + str(p_value) + "."

    return final_payload


def train_market(train_data):
    """
    Train artificial prediction market agents for the supplied dataset

    :param train_data: Dataset to be used for training
    :return: Training status
    """
    config_file = open(r'app_config.yaml')
    config = yaml.safe_load(config_file)
    weight_write_path = config['APP']['TRAINING']['WEIGHTS_PATH']

    # Write dataset to local directory
    with open('./dataset/{0}'.format(train_data.filename), 'wb') as f:
        shutil.copyfileobj(train_data.file, f)
        f.close()

    data = pd.read_csv('./dataset/{}'.format(train_data.filename))
    # Remove identifier fields
    id_fields = config['APP']['TRAINING']['IDENTIFIER_FIELDS']
    data = data.drop(id_fields, axis=1)

    # Initialization
    set_config = InitialConfiguration(data.values[:, :])
    dataset_init_path = './dataset/{}_init.csv'.format(train_data.filename)
    set_config(normalize=False, output_path=dataset_init_path)
    data_with_init = pd.read_csv(dataset_init_path, header=None)
    feature_vectors = data_with_init.iloc[:, :-3].values.tolist()
    labels = data_with_init.iloc[:, -3].values.tolist()
    start_config = data_with_init.iloc[:, -2:].values.tolist()
    start = np.array(start_config)

    train_feature_vectors = np.array(feature_vectors)
    train_labels = np.array(labels)

    # TODO: Currently the market has its own configuration file. Consider moving everything to the single
    #  consolidated configuration: app_config.yaml

    # Train agents -> Agent weights path set in Market/config.py
    # Verify if the specified directory to save weights to exists, if not create
    if not path.exists(output_folder_location):
        makedirs(output_folder_location)

    try:
        train_ga(train_feature_vectors, train_labels, start_config=start)
    except Exception as e:
        print(e)
        return {"Status": "Training failure, check error trace in backend"}
    return {"Status": "Agents trained based on the provided dataset"}
