"""
Common utility function definitions are composed in this file for the Feature Pipeline
in addition to limited lookup values.
"""

from collections import namedtuple
import pandas as pd
import csv
import re
import string

p_val_sign = {
    '<': -1,
    '=': 0,
    '>': 1
}

tamu_select_features = ["Venue_Citation_Count", "Venue_Scholarly_Output", "Venue_Percent_Cited", "Venue_CiteScore",
                        "Venue_SNIP", "Venue_SJR", "Venue_subject", "Venue_subject_code", "avg_pub", "avg_hidx", "avg_auth_cites",
                        "avg_high_inf_cites", "sentiment_agg"]


def remove_accents(text: str):
    text = re.sub('[âàäáãå]', 'a', text)
    text = re.sub('[êèëé]', 'e', text)
    text = re.sub('[îìïí]', 'i', text)
    text = re.sub('[ôòöóõø]', 'o', text)
    text = re.sub('[ûùüú]', 'u', text)
    text = re.sub('[ç]', 'c', text)
    text = re.sub('[ñ]', 'n', text)
    text = re.sub('[ÂÀÄÁÃ]', 'A', text)
    text = re.sub('[ÊÈËÉ]', 'E', text)
    text = re.sub('[ÎÌÏÍ]', 'I', text)
    text = re.sub('[ÔÒÖÓÕØ]', 'O', text)
    text = re.sub('[ÛÙÜÚ]', 'U', text)
    text = re.sub('[Ç]', 'C', text)
    text = re.sub('[Ñ]', 'N', text)
    return text


def strip_punctuation(text: str):
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    text = re.sub(regex, "", text)
    text = text.strip()
    return text


def read_darpa_tsv(file):
    """
    Generator to iterate over metadata for COVID sprint evaluation papers

    :params file: The metadata file as supplied by OSF
    """
    df = pd.read_csv(file, sep="\t")
    for index, row in df.iterrows():
        try:
            yield {"title": row['title_CR'], "pub_year": row['pub_year_CR'], "doi": row['DOI_CR'],
                   "ta3_pid": row['ta3_pid'], "pdf_filename": row['pdf_filename'], "claim4": row['claim4_inftest']}
        except KeyError:
            ta3_pid = row['pdf_filename'].split()[-1]
            yield {"title": row['title_CR'], "pub_year": row['pub_year_CR'], "doi": row['DOI_CR'],
                   "ta3_pid": ta3_pid, "pdf_filename": row['pdf_filename'], "claim4": row['claim4_inftest']}


def elem_to_text(elem, default=''):
    """
    Read text from XML element. Defaults to empty string if unavailable

    :param elem: XML element with text that needs reading
    :param default: Default values if elem does not exist
    :return: Text contained in element
    """
    if elem:
        return elem.getText()
    else:
        return default


def csv_writer(filename, append=False):
    """
    Creates a CSV writer object for the supplied file | Used to create dataset CSVs

    :param filename: File to write out to
    :param append: Open file in append mode or create mode. False creates a new file or overwrites an existing one.
    :return: CSV writer object
    """
    if append:
        writer = csv.writer(open(filename, 'a', newline='', encoding='utf-8'))
    else:
        writer = csv.writer(open(filename, 'w', newline='', encoding='utf-8'))
    return writer


def csv_write_field_header(writer, header):
    """
    Write header with column names into CSV

    :param writer: CSV writer object create using csv_writer
    :param header: List containing the names of the columns to be written into the CSV
    """
    writer.writerow(header)


def csv_write_record(writer, record, header):
    """
    Write dict based record into CSV in order

    :param writer: CSV writer object create using csv_writer
    :param record: Datapoint/feature vector to be written into the CSV
    :param header: Matches elements by the elements in the header
    """
    nt_record = namedtuple('dis_features', header)
    print(nt_record)
    sim_record = nt_record(**record)
    writer.writerow(list(sim_record))


def select_keys(input_data, projection=None):
    """
    Read selected values from dictionary

    :param input_data: Full set of key:values (dictionary)
    :param projection: List of keys that need to be read and returned for the desired subset
    :return: Dictionary containing subset of keys as defined in projection
    """
    output_projection = {}
    for key in projection:
        try:
            output_projection[key] = input_data[key].values[0]
        except:
            output_projection[key] = 0
    return output_projection
