# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 00:35:43 2022

@author: weixi
"""


import pandas as pd
from pandas.core.frame import DataFrame
import re
import os


file =pd.read_csv(r'D:\Data\score_pileline_app\metadata\data_yang.csv')

y = file['y'].tolist()      #dataframe => series => list
doi = file['DOI_CR'].tolist()
pdf_filename = file['pdf_filename'].tolist()
title = file['title_CR'].tolist()

doi_dict ={}
title_dict = {}
for i in range(0, len(doi)):
    doi_dict[str(pdf_filename[i])] = doi[i]
    title_dict[str(pdf_filename[i])] = title[i]
    
    
a = doi_dict['18']
b = title_dict['18']