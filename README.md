## Feature extraction Pipeline

The pipeline is designed to extract features from scholarly work. Given a paper in PDF format, it extracts
features and writes them into a CSV file. The table below summarizes the features currently built-in to the pipeline.

| Feature        | Description     |   
| -------------  |:-------------   |
| doi            |Unique identfier for each record |
| title          | Title of the paper |
| num_citations  | Number of citations for paper |
| author count   |Number of authors |
| sjr            | Scientific Journal Rank |
| u_rank         | University Rank - Based on author affiliiation |
| self citations | Self-citations/total number of references |
| subject        | A subject code based on the high level categorization of the paper based on Semantic Scholar's subject classification |
| subject code   | Specific subject code class based on Semantic Scholar's subject classification |
| citation velocity | Measures how many citations a given paper has picked up over the past three years |
| influential_citation_count | Number of citations received from papers that are classified as "infuential" on Semantic Scholar. |
| references_count | Number of references |
| open_access_flag | Whether the paper is available via open access on the web |
| normalized citations | Derived feature based on num citations and year of publication |
| num hypo tested | Count of the number of statistical p-values found in the paper |
| real_p | The smallest significant p-value extracted |
| real_p_sign | Class based feature based on the sign in the p-value expression |
| p_val_range | Smallest extracted p-value subtracted from the largest extracted p-value |
| num_significant | Number of significant p-values extracted (<= 0.05) |
| sample_size | Maximum value from the sample sizes found or calculated from the pdf |
| extend_p | Boolean that is set in case the p-value features are derived from p-values not found to be associated/matched with a test during extraction |
| funded | Based on acknowledgements extraction. If one of the entities acknowledged is an organization, set to 1 |
| Venue CiteScore | The metric is a standard to help measure citation impact for journals, book series, conference proceedings and trade journals |
| Venue SNIP | The SNIP indicator measures the average citation impact of the publications of a journal, using Scopus data. |
| Venue_Scholarly_Output | Scholarly output defines the total count of research output, to represent productivity This feature is calculated as the sum of documents published in a certain venue in the 3 years prior to the current year |
| Venue_Percent_Cited | This is calculated as the proportion of documents that have received at least 1 citation |
| Venue_Citation_Count | This feature is calculated as the number of citations received in one year for the documents published in the previous 3 years |
| author_count | The total number of authors of the target paper |
| avg_pub | The average number of publications of all authors of the target paper |
| avg_hidx | The average h-index of all authors of the target paper |
| avg_high_inf_cites | The average number of highly influential citations aggregated across all authors of the paper |
| avg auth cites | The average number of citations of all authors |
| Subjectivity | Scores how subjective the abstract is. Value ranges between 0,1 with 1 being very subjective |
| Ease_of_Reading | Scores how easy it is to read abstract. We use Flesch Reading Ease as the indicator |
| Sentiment | Output the sentiment of the abstract. Value ranges between positive to negative |


#### PIPELINE SETUP SYSTEM REQUIREMENTS:

1) Make sure to install the Java SDK 11. (JVM 11)
2) The Python version currently supported by the feature pipeline is Python 3.6.13


### The overall feature pipeline is consist of two parts:

1) Preprocessing the input research papers PDFs

2) Generating features using the processed documents.

PDFs processed using GROBID and pdf2text. This preprocessing can be done using the pipeline or independently before
extracting features.

## Preprocessing PDFs
PDFs are processed using GROBID and pdf2text. This preprocessing can be done using the pipeline or independently before
extracting features. In both cases, it is required to have a working GROBID and pdf2text installation. Please follow the stepwise instructions for the proper installation of GROBID and pdf2text:


# GROBID

GROBID (version 0.7.0) installation guide is available here: [GROBID](https://grobid.readthedocs.io/en/latest/Install-Grobid/)

1) Get latest stable release
```
mkdir Grobid
```
```
cd Grobid
```
```
wget https://github.com/kermitt2/grobid/archive/0.7.0.zip
unzip 0.7.0.zip
```
2) Build GROBID with Gradle

```
./gradlew clean install
```

3) Start the server with Gradle (under the grobid/ main directory)
```
./gradlew run
```

NOTE: You can check whether the service is up and running by opening the following URL: http://yourhost:8070/api/version which will return you the current version. Another way is going to - http://yourhost:8070/api/isalive which will return true/false whether the service is up and running.


4) Configuring the GROBID service

```
GROBID needs to be running on a separate terminal and it runs by default on port 8070 (grobid_port). Please update the grobid_server in the "score->darpa-market->score_app->feature pipeline->grobid_client->config.json" based on your system's localhost IP.
```

While preprocessing with GROBID, it is required to convert using full text mode i.e., **/api/processFulltextDocument**,
please refer to GROBID documentation for more details.

# PDF2Text
[PDF2Text](https://linux.die.net/man/1/pdftotext) parse the paper PDFs into files to plain text (version 4.03).


1) Get the XpdfReader toolkit
```
!wget --no-check-certificate https://dl.xpdfreader.com/xpdf-tools-linux-4.03.tar.gz
!tar -xvf xpdf-tools-linux-4.03.tar.gz && sudo cp xpdf-tools-linux-4.03/bin64/pdftotext /usr/local/bin
```

2) Update the path of pdftotext in the “process_docs.py”. See line 32:
```
“windows_pdftotext_path = '../xpdf-tools-linux-4.03/bin64/pdftotext'”
```

NOTE: Make sure to install Poppler on the system (python)

# Download the Score source code:
```
mkdir Score
cd Score
git clone https://git.psu.edu/darpascore/darpa-market
```
NOTE: Under Score folder, create a data folder and another database folder under data (for storing .db files).
Update the path (database_path) value to the database folder in "data_processor.py".
```
cd Score
mkdir data
mkdir data/database
```


Once GROBID is installed and running, and pdf2text is installed along with the latest score source code, we can use the pipeline to
preprocess the PDF files using the below command:

```
python process_docs.py --mode  process-pdfs  --pdf_input DIR_TO_PDFs -out OUTPUT_DIR
```

* DIR_TO_PDFs: path to the folder containing the PFDs
* OUTPUT_DIR: path to the folder to store GROBID output

NOTE: The output of the pdf2text will be saved in the "DIR_TO_PDFs".

Alternatively, one can process them separately without using the pipeline.

Note that the docker implementation comes with grobid built-in along with native pdftotext support in the application
container. Datasets are currently generated using the shell prompt shown above.


## Running the pipeline feature extraction

Once the PDFs are processed using GROBID and pdf2text, we can run a pipeline for feature extraction:

IMP_NOTE: Create a conda environment `conda create --name score python=3.6` (See Conda_config for installing required dependencies)


NOTE: Update the path values for the following in the code:

1) 'journal_dictionary.csv' in the elsevier_api.py
2) 'journal_dictionary.pkl' in the extractor.py and journal_dict.py
3) 'uni_rank.pickle' in the extractor.py


Now, run the pipeline using the below command:

`python process_docs.py -out PROCESSED_GROBID_FILES -in TEXT_FILES -m generate-train  -csv OUTPUT_DIR -l 0_or_1`

-out: path to preprocessed PDF files in tei.xml format(grobid output)

-in: path to preprocessed PDF files in txt format(output of pdf2text)

-m: generate-train mode

-csv: csv output directory

-l: label 'y' (set to 0 or 1)

For more details, refer to process_docs.py.

OUTPUT: csv will be generated at: `../train.csv`


## Integrating new theory features

 IMP_NOTE: Create a conda environment `conda create --name flair python=3.6` (See Conda_configs/theory_count.txt for installing required dependencies)

 Create an environment: `conda activate flair`

 Install the flair: `pip install flair`

Need to download the theory model from Google drive: https://drive.google.com/file/d/15CgrLfxLPjZHmAGDQgEu8r-hASOlRugG/view?usp=sharing

Inside the theory.py update following paths:

Add the model path here: `model = SequenceTagger.load('../resources/taggers/example-ner/final-model.pt')`

Add the path to meta data csv: `data = pd.read_csv('../data.csv')`

Add the path to 40 feature train csv: `csv_path = '../train.csv'`

Now, generate and add theory count feature, run: `python3 theory.py`

OUTPUT: csv will be generated at: `../train_41.csv`

## Integrating new negation count features

Install following python librabies:

`pip install spacy==3.0.1`
`pip install pytest`
`pip install negspacy`

Inside the negation.py update following paths:

Add the path to meta data csv: `raw_data = "../data.csv")`

Add the path to 41 feature train csv file `csv_path = '../train_41.csv'`

OUTPUT: csv will be generated at: `../train_41_45.csv`
This csv contains 45 features for training.


# Bugs fixed in this version:
* p_value.py error fixed (null Z-DISTRIBUTION handling)

* process_docs.py error fixed (sentiment API abstract input length edge case handling)


### Project Structure
Important files for reference:

| File        | Description     |   
| ------------- |:------------- |
| process_docs.py     | code execution starts here, there are 2 main modes (1) Preprocess (2) generate feature set |
| extractor.py      | grobid output gets torn down into various features and extracted information is used to call elsevier/crossref/semantic scholar api     |
| elsevier.py | Output from elsevier api/crossref/semantic scholar gets parsed and returned |
| XIN.py | acknowlegement section is processed to identify funding information |

**NOTE:** Elsevier api key may expire after certain number of hits. In case of batch processing, it is better to update api key details from elsevier developer [portal](https://dev.elsevier.com). Check the same for semantic scholar api.


**NOTE** Place the citation sentiment model under Feature_Pipeline/tamu_features/rec_model/pytorch_model.bin from the [link](https://drive.google.com/file/d/1Yd_x-65bCqu8kJlo6QaIltoNv2CCEU0w/view?usp=sharing)


**NOTE** Place the citation sentiment model under Feature_Pipeline/tamu_features/rec_model/pytorch_model.bin from the [link](https://drive.google.com/file/d/1Yd_x-65bCqu8kJlo6QaIltoNv2CCEU0w/view?usp=sharing)
