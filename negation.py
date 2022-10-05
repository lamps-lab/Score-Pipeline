# -*- coding: utf-8 -*-
"""
Negation detection using Negspacy
"""
# pip install -r requirement.txt
# python -m spacy download en

import spacy
from negspacy.negation import Negex
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline

def get_preprocess(raw_data):
  df= pd.read_csv(raw_data)
  return df

raw_data="/content/data.csv"    #testing a file
df=get_preprocess(raw_data)

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("negex", config={})

def sen(sentence):
  doc = nlp(sentence)
  countT=0
  countF=0
  for e in doc.ents:
    if e._.negex==True:
      countT=countT+1
    elif e._.negex==False:
      countF=countF+1
      # listL=[]
  if countT==0 and countF==0:
    return 0
  elif countT>=1:
    return 1
  else:
    return 0
    # print(e.text, e._.negex)

use=df.index

paperid=df['paper_id'].to_list()

listD=[]
for i in use:
  value=sen(df.coded_claim4[i]) # coded_claim4 is the column name needed to check for negation
  listD.append(value)

listC=[]
for i in use:
  value=sen(df.coded_claim3b[i])
  listC.append(value)

listB=[]
for i in use:
  value=sen(df.coded_claim3a[i])
  listB.append(value)

listA=[]
for i in use:
  value=sen(df.coded_claim2[i])
  listA.append(value)

output = pd.DataFrame(
    {'paper_id': paperid, 'Claim4': listD,
     'Claim3b': listC,
     'Claim3a': listB,
     'Claim2':listA
    })

# output.to_csv('Negation_Output.csv')

def add_to_csv(l_dict):

    csv_path = '/content/400_train_41.csv'      # path to the 40 feature train csv

    df = pd.read_csv(csv_path)
    l = df.shape[1]
    df.insert(l, "Claim4", 0)     # Adding new column for theory count, default value '0'
    df.insert(l+1, "Claim3a", 0)     # Adding new column for theory count, default value '0'
    df.insert(l+2, "Claim3b", 0)     # Adding new column for theory count, default value '0'
    df.insert(l+3, "Claim2", 0)     # Adding new column for theory count, default value '0'


    for i in range(len(l_dict)):
        # print(a)
        try:

            index = df[df['paper_id'] == l_dict['paper_id'][i]].index.tolist()[0]  # get the index of paper in csv
        # print(index)
            df['Claim4'][index] = l_dict['Claim4'][i]              # append the respective negation for the paper
            df['Claim3a'][index] = l_dict['Claim3a'][i]              # append the respective negation for the paper
            df['Claim3b'][index] = l_dict['Claim3b'][i]              # append the respective negation for the paper
            df['Claim2'][index] = l_dict['Claim2'][i]              # append the respective negation for the paper

        except:
            print(l_dict['paper_id'][i])

        xx = csv_path.split('.')
        df.to_csv(xx[0] + '_45.csv')



add_to_csv(output)
print('Adding feature to CSV')
print('==============================================================')
print('Done')
print('==============================================================')
