# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 00:50:09 2021

@author: weixin & rajratnpranesh
"""


from flair.data import Sentence
from flair.training_utils import EvaluationMetric
from flair.data import Corpus
from flair.data_fetcher import NLPTaskDataFetcher, NLPTask
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, FlairEmbeddings, TransformerWordEmbeddings
from typing import List
from flair.models import SequenceTagger
import json
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'


    #
    #
    # parser = argparse.ArgumentParser(description="Adding scifact features to csv")
    # parser.add_argument("-in_csv", "--csv_input", help="feature csv path")
    # parser.add_argument("-out", "--csv_out",  help="path to save final csv")
    # parser.add_argument("-in_scif", "--json_input", help="scifact json path")
    #
    #

import pandas as pd
import pickle
import re
import string
import os
import subprocess
def remove_accents(text):
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

# load the model you trained
model = SequenceTagger.load('/Users/rajratnpranesh/xin/resources/taggers/example-ner/final-model.pt')
#download the model from Google drive: https://drive.google.com/file/d/15CgrLfxLPjZHmAGDQgEu8r-hASOlRugG/view?usp=sharing

def get_dict(js):

    data_dict = js
    t_dic = dict() # key = paper id, value =  no. of theory

    theory_dic = dict()
    ccc = 0


    for key in data_dict:  #  data_dict[key] is a list of sentences (the claimzone), key is the name of the claimzone
        claim_sentence = data_dict[key]
            #print(claim_sentence)
            #print('===')

            # create example sentence
        sentence = Sentence(claim_sentence)
        # print(sentence)

        # predict tags and print
        model.predict(sentence)

        output = sentence.to_tagged_string()
        # print(output)
        TH = []
        ccc = ccc + 1
        for string1 in sentence.get_spans('ner'):
            # print(string1)
            string1 = str(string1)  # span object has no attribute split
            strint1_split = string1.split('   ')

            label1 = strint1_split[1].split(': ')  #['[− Labels', 'ASPECT (0.9998)]']
            label = label1[1].split(' (')[0]     #ASPECT
            # print(label)
            entity = strint1_split[0].split(': ')[1]   #"front perspective view"
            entity = entity.split('"')    #['', 'front perspective view', '']
            entity = entity[1]          #front perspective view
            # print(entity)
            if label == "TH":
                TH.append(entity)
                # print(TH)
            else:
                TH = TH

        theory_dic[key]=TH
        # print(theory_dic)
        # print('NEW NEW NEW added @@@@@@@@@@ # #  #  # #  # # ###########')
    # print(theory_dic)
    # print("ttttTH:", TH)
    # print('=================================================================')
        p_id = key.split('_')[0]  # Get the paper id
        if p_id not in t_dic:
            t_dic[p_id] = TH      # append all the theory for each paper
        else:
            t_dic[p_id].append(TH)
        print(ccc)
        # p_id = key.split('_')[0]  # Get the paper id
        # if p_id not in t_dic:
        #     t_dic[p_id] = TH      # append all the theory for each paper
        # else:
        #     t_dic[p_id].append(TH)
        # print(t_dic[p_id],TH)
    return t_dic

def get_unq(t_dict):
    s_list = []
    all_list = []
    f_dic =  dict()   #key: paper paper_TITLE    value: theory count

    for k,v in t_dict.items():
        pp_id = k                    #paper id
        s_list = []
        all_list = []
        for u in v:                 # one list of theory for a paper
            if isinstance(u, str):
                s_list.append(u)
        for uu in v:
            if isinstance(uu, list):
                all_list.append(uu)

        all_list.append(s_list)    # all list of theory in a parent list
        flat_list = [item for sublist in all_list for item in sublist]  # single list for list of list
        unq_theory = set(flat_list)   # all the unique theory in one paper
        count_ut = len(unq_theory)    # count of unque theory

        if pp_id not in f_dic:
            f_dic[pp_id] = count_ut

    return f_dic
#



# main code

data = pd.read_csv('/Users/rajratnpranesh/xin/data.csv')  # path to meta data csv

god_dic = {}

for i in range(len(data)):

  paper_id = data['paper_id'][i]

  claim2_coded = data['coded_claim2'][i]
  claim2_coded = remove_accents(claim2_coded)
  c2 = paper_id + '_C2'
  if c2 not in god_dic:
      god_dic[c2] = claim2_coded


  claim3a_coded = data['coded_claim3a'][i]
  claim3a_coded = remove_accents(claim3a_coded)
  c3a = paper_id + '_C3a'
  if c3a not in god_dic:
      god_dic[c3a] = claim3a_coded

  claim3b_coded = data['coded_claim3b'][i]
  claim3b_coded = remove_accents(claim3b_coded)
  c3b = paper_id + '_C3b'
  if c3b not in god_dic:
      god_dic[c3b] = claim3b_coded


  claim4_coded = data['coded_claim4'][i]
  claim4_coded = remove_accents(claim4_coded)
  c4 = paper_id + '_C4'
  if c4 not in god_dic:
      god_dic[c4] = claim4_coded

#
# print(god_dic)





#
#
def add_to_csv(l_dict):

    csv_path = '/Users/rajratnpranesh/xin/train.csv'      # path to the 40 feature train csv

    df = pd.read_csv(csv_path)
    df.insert(df.shape[1], "theory_count", 0)     # Adding new column for theory count, default value '0'

    for a,b in l_dict.items():
        # print(a)
        try:

            index = df[df['paper_id'] == a].index.tolist()[0]  # get the index of paper in csv
        # print(index)
            df['theory_count'][index] = b                   # append the respective theory count for the paper

        except:
            print(a)

        xx = csv_path.split('.')
        df.to_csv(xx[0] + '_41.csv')                        # overweite the csv with new csv with added column



# MAIN FUNCTION CALLS

res = get_dict(god_dic)
print('Getting Theories')
result = get_unq(res)
print('Getting Unique Theories')
add_to_csv(result)
print('Adding feature to CSV')
print('==============================================================')
print('Done')
print('==============================================================')
