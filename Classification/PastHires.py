"""
Q1. (Create a program that fulfills the following specification.)
PastHires.csv

Here, we are building a decision tree to check if a person is hired or not based 
on certain predictors. scikit-learn needs everything to be numerical for decision 
trees to work.
So,use any technique to map Y,N to 1,0 and levels of education to some scale of 0-2.

1.Build and perform Decision tree based on the predictors and see how 
  accurate your prediction is for a being hired.

2.Now use a random forest of 10 decision trees to predict employment of 
  specific candidate profiles:

a.Predict employment of a currently employed 10-year veteran, previous employers 4,
  went to top-tire school, having Bachelor's Degree without Internship.
b.Predict employment of an unemployed 10-year veteran, ,previous employers 4, 
  didn't went to any top-tire school, having Master's Degree with Internship.
"""

import pandas as pd
import numpy as np

# This is  a regression problem
dataset = pd.read_csv('All CSV/PastHires.csv')  
dataset.info()
dataset.head()

# Finding missing data
dataset.isnull().any(axis=0)

features = dataset.drop('Hired', axis=1).values  
labels = dataset['Hired'].values

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

# Encoding categorical data
labelencoder = LabelEncoder()
for i in [1,4,5]:
    features[:, i] = labelencoder.fit_transform(features[:, i])
print(features)

# encoding for column 'level of education'
columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [3])], remainder='passthrough')
features = np.array(columnTransformer.fit_transform(features), dtype = np.float32)
features = features[:,1:]   ## removing redundant column
print(features)
"""
BS: 0, 0
MS: 1, 0
Phd:0, 1
"""

from sklearn.tree import DecisionTreeClassifier  
classifier = DecisionTreeClassifier()  
classifier.fit(features, labels)
labels_pred = classifier.predict(features) 

# Comparing the predicted and actual values
my_frame= pd.DataFrame({'Actual':labels, 'Predicted':labels_pred})
print(my_frame)

# For classification tasks some commonly used metrics are confusion matrix
from sklearn.metrics import confusion_matrix  
cm = confusion_matrix(labels, labels_pred)
'''
[[4 0]
 [0 9]]
'''
# Model Score = 100.00 times out of 100 model prediction was RIGHT
print( (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[1][1] + cm[0][1] + cm[1][0])) 

################################################################################

'''
Random forest of 10 decision trees
'''

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()  
features = sc.fit_transform(features)  
#features_test = sc.transform(features_test) 

#train the model
from sklearn.ensemble import RandomForestClassifier
classifier2 = RandomForestClassifier(n_estimators=10, random_state=0)  
classifier2.fit(features, labels)  

labels_pred = classifier2.predict(features) 
# Comparing the predicted and actual values
my_frame= pd.DataFrame({'Actual':labels, 'Predicted':labels_pred})
print(my_frame)

#Evaluate the algo
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(labels,labels_pred))  
print(classification_report(labels,labels_pred))  
print(accuracy_score(labels, labels_pred))  ## 100%

'''
[[4 0]
 [0 9]]
'''

#####################

"""
1.Predict employment of a currently employed 10-year veteran, previous employers 4,
  went to top-tire school, having Bachelor's Degree without Internship.
"""

x= np.array([10,1,4,'BS',1,0])
x = x.reshape(1,-1)  ## convrting 1d to 2d array
x = np.array(columnTransformer.transform(x), dtype = np.float32)
x = x[:,1:]

pred = classifier.predict(x)
print('Hired' if pred[0]=='Y' else 'Rejected')


## by random forest method
x = sc.fit_transform(x)  
pred = classifier2.predict(x)
print('Hired' if pred[0]=='Y' else 'Rejected')

"""
2.Predict employment of an unemployed 10-year veteran, ,previous employers 4, 
  didn't went to any top-tire school, having Master's Degree with Internship.
"""

y= np.array([10,0,4,'MS',0,1])
y = y.reshape(1,-1)  ## convrting 1d to 2d array
y = np.array(columnTransformer.transform(y), dtype = np.float32)
y = y[:,1:]

pred = classifier.predict(y) 
print('Hired' if pred[0]=='Y' else 'Rejected')


## by random forest method
y = sc.fit_transform(y)  
pred = classifier2.predict(y) 
print('Hired' if pred[0]=='Y' else 'Rejected')





