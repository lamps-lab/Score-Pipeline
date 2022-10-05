# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 14:42:01 2022

@author: weixi
"""

import pandas as pd
from pandas.core.frame import DataFrame
import re
import os


file =pd.read_excel(r'D:\Data\score_pileline_app\metadata\metadata_yang.xlsx')

selected_data = file[['ID', 'DOI', 'Study Title (O)', 'Replicate_Binary']] # a list

# list to dataframe
selected_df = DataFrame(selected_data)   # from pandas.core.frame import DataFrame

df2 = selected_df.rename({'ID': 'pdf_filename', 'DOI': 'DOI_CR', 'Study Title (O)': 'title_CR', 'Replicate_Binary': 'y'}, axis='columns')

b = df2['pdf_filename'][2]
print(type(b))

df2.to_csv('data_yang_v1.csv', index= False)





            