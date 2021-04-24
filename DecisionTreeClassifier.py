#Decision Tree Classifier

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

import numpy as np
import pandas as pd
import os

os.chdir("File Path")

def split_dataset(dataset, train_percentage, feature_headers, target_header):
       train_x, test_x, train_y, test_y = train_test_split(dataset[feature_headers], dataset[target_header], train_size=train_percentage)
       return train_x, test_x, train_y, test_y

def decision_tree_classifier(features, target):
      clf = DecisionTreeClassifier(criterion = "gini")
      clf.fit(features, target)
      return clf

headers = ['Industry', 'Deal_value','Weighted_amount', 'Pitch','Lead_revenue','Fund_category','Geography', 'Designation','Lead_source','Level_of_meeting','Last_lead_update','Internal_POC','Resource','Success_probability']

def main():
       dataset = pd.read_csv('traincat2.csv')
       dataset = dataset.astype(int);
       train_x, test_x, train_y, test_y = split_dataset(dataset, 0.6, headers[0:13], headers[13])
       trained_model = decision_tree_classifier(train_x, train_y)
       predictions = trained_model.predict(test_x)
       print ("Train Accuracy :: ", accuracy_score(train_y, trained_model.predict(train_x)))
       print ("Test Accuracy  :: ", accuracy_score(test_y, predictions))
       print ("Trained model :: ", trained_model)

if __name__ == "__main__":
     main()
