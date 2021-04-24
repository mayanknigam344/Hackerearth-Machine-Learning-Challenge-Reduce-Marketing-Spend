#Preprocessing of data

import numpy as np
import pandas as pd
import os

os.chdir("C:/Users/mayan/OneDrive/Desktop/ML_Project/Datasets/dataset")
data = pd.read_csv('train.csv')
data = data.drop(labels = ['Deal_title', 'Lead_name' ,'Date_of_creation'] , axis = 1)
data = data.drop(labels = ['Contact_no' , 'Location' , 'POC_name' ,'Lead_POC_email' , 'Internal_rating'] , axis = 1)
data = data.dropna()
data = data[~data.isin(['?'])]
data = data.dropna()
data.to_csv('train1.csv')

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df = pd.read_csv('train1.csv')
cols = ['Fund_category', 'Geography','Internal_POC', 'Level_of_meeting','Lead_source','Pitch','Designation', 'Lead_revenue','Industry','Hiring_candidate_role','Last_lead_update','Resource']
df[cols] = df[cols].apply(le.fit_transform)
df.to_csv('traincat.csv')

df['Deal_value'] = df['Deal_value'].str.replace('$', '')
df['Weighted_amount'] = df['Weighted_amount'].str.replace('$', '')
df.to_csv('traincat1.csv')
pf = pd.read_csv('traincat1.csv')
pf['Deal_value'] = pd.cut(pf['Deal_value'], bins=10, labels=np.arange(10), right=False)
pf['Weighted_amount'] = pd.cut(pf['Weighted_amount'], bins=10, labels=np.arange(10), right=False)
pf.to_csv('traincat2.csv')


