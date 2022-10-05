# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 14:42:01 2022

@author: weixi
"""

import pandas as pd
from pandas.core.frame import DataFrame
import re
import os


file =pd.read_excel(r'D:\Data\score_pileline_app\metadata\metadata_all_issn.xlsx')

selected_data = file[['ID', 'DOI', 'Study Title (O)', 'Replicate_Binary', 'ISSN']] # a list

# list to dataframe
selected_df = DataFrame(selected_data)   # from pandas.core.frame import DataFrame

df2 = selected_df.rename({'ID': 'pdf_filename', 'DOI': 'DOI_CR', 'Study Title (O)': 'title_CR', 'Replicate_Binary': 'y'}, axis='columns')

#for index, row in df2.iterrows():   
#    if pd.isnull(row['DOI_CR']):
#        df2 = df2.drop(index)

b = df2['y'][2]
#print(type(b))

new_pdfname =[0]*len(df2['pdf_filename'])
for index, row in df2.iterrows():
    new_pdfname[index] = int(row['pdf_filename'])
    #df2['pdf_filename'][index] = 8
    #print(df2['pdf_filename'][index])
    #print(type(df2['pdf_filename'][index]))
    

df2 = df2.drop('pdf_filename', axis=1)   # drop column
df2['pdf_filename'] = new_pdfname       # add column
print(type(df2['pdf_filename'][2]))
           
           
for index, row in df2.iterrows():
    #print(index)
    if type(row['DOI_CR']) ==str:  # note: not 'str'
        #print(row['DOI'].split(' ')[0])
        if row['DOI_CR'].split(' ')[0]=='duplicate':
            #print(row['DOI_CR'].split(' ')[0])
            #print(index)
            df2 = df2.drop(index)
            
    elif pd.isnull(row['DOI_CR']):
        df2 = df2.drop(index) 
        

b = df2['pdf_filename'][2]
print(type(b))     # <class 'numpy.float64'>
b1= int(b)
print(type(b1))     # <class 'int'>

     
for index, row in df2.iterrows():
     #print(row['ISSN'])
     if pd.isnull(row['ISSN']):
         print(index)
     else:
         df2['ISSN'][index] = row['ISSN'].replace('-','')
         #print(row['ISSN'])
         #print(index)


           
df2.to_csv('data_yang.csv', index= False)





            