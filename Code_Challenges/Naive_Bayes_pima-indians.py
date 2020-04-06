"""
Q1. dataset: pima-indians-diabetes.csv

Perform both Multinomial and Gaussian Naive Bayes classification 
after taking care of NA values (maybe replaced with zero in dataset)

Calculate accuracy for both Naive Bayes classification model.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing dataset
data = pd.read_csv("All CSV/pima-indians-diabetes.csv")
print(data.shape)
data.info()

# checking missing values
data.isnull().any(axis=0)

# Split dataset in training and test datasets
from sklearn.model_selection import train_test_split

features_train, features_test =\
train_test_split(data, test_size=0.5, random_state=0)
# we are passing full data as features and no labels are passed

from sklearn.naive_bayes import GaussianNB, MultinomialNB

"Gaussian Naive Bayes"
gnb = GaussianNB()
used_features =[
    "6",
    "148",
    "72",
    "35",
    "0",
    "33.6",
    "0.627",
    "50"
]  # "Survived" is the column for labeleling 

# Train classifier
gnb.fit(
    features_train[used_features].values,  # features are passed
    features_train["1"].values      # labels is passed
)

labels_pred = gnb.predict(features_test[used_features])

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score  
cm_gnb = confusion_matrix(features_test["1"], labels_pred)
print(cm_gnb)

# Score
print (accuracy_score(features_test["1"], labels_pred)) # 76.04%
##########################

"Multinomial classification"
mnb = MultinomialNB()

# Train classifier
mnb.fit(
    features_train[used_features].values,
    features_train["1"].values
)
labels_pred = mnb.predict(features_test[used_features])

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_mnb = confusion_matrix(features_test["1"], labels_pred)
print(cm_mnb)

# Score
print (accuracy_score(features_test["1"], labels_pred)) # 63.54%








