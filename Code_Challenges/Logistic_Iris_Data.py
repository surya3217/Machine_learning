"""
The Iris flower data set or Fisher's Iris data set is a multivariate data set 
introduced by the British statistician and biologist Ronald Fisher in his 1936 paper.
The use of multiple measurements in taxonomic problems as an example of linear 
discriminant analysis.

The data set consists of 50 samples from each of three species of Iris (Iris setosa,
Iris virginica and Iris versicolor). Four features were measured from each sample: 
the length and the width of the sepals and petals, in centimetres. Based on the 
combination of these four features, Fisher developed a linear discriminant model to
distinguish the species from each other.
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# We import this dataset already in sklearn, and perform logisitic regression based on
# petals and sepals properties to classify the iris species
from sklearn.datasets import load_iris

iris = load_iris() # Loading Iris Dataset
print (iris.feature_names)  # Column Names for the iris dataset

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
        train_test_split(iris.data, iris.target, test_size = 0.25, random_state = 0)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
logClassifier = LogisticRegression(random_state=0)
logClassifier.fit(features_train, labels_train)

# Printing Score for the Regression Model
print (logClassifier.score(features_train,labels_train)*100 ) ## 94.64 %

# Predicting the Test set results
labels_pred = logClassifier.predict(features_test)
print (labels_pred)

'''
[[13  0  0]
 [ 0 11  5]
 [ 0  0  9]]
'''
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels_test, labels_pred)
print(cm)


