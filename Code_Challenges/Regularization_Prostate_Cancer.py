"""
Code Challenge 01: 
(Prostate Dataset) Prostate_Cancer.csv
Load the dataset from given link: 
pd.read_csv("http://www.stat.cmu.edu/~ryantibs/statcomp/data/pros.dat", delimiter =' ')

This is the Prostate Cancer dataset. Perform the train test split before you apply the model.

(a) Can we predict lpsa from the other variables?
(1) Train the unregularized model (linear regressor) and calculate the mean squared error.
(2) Apply a regularized model now - Ridge regression and lasso as well and check the mean squared error.

(b) Can we predict whether lpsa is high or low, from other variables?
"""

#Import libraries
import pandas as pd  
import numpy as np  

#import database
dataset = pd.read_csv('All CSV/Prostate_Cancer.csv') 
dataset.info()
dataset.head()

# Finding missing data
dataset.isnull().any(axis=0)

## Handle missing values
dataset['compactness']= dataset['compactness'].fillna(np.mean(dataset['compactness']))
dataset['fractal_dimension']= dataset['fractal_dimension'].fillna(np.mean(dataset['fractal_dimension']))
dataset.info()

features = dataset.iloc[:, 2:].values  
labels = dataset.iloc[:, 1]

## label encoding
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
labels = labelencoder.fit_transform(labels)
print(labels)

labels= np.array( labels, dtype= np.float64)


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
features_train = sc.fit_transform(features_train)
features_test = sc.transform(features_test)

"""
Unregularized model - Linear regressor
Regularized model - Ridge regression and lasso
"""
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso 
from sklearn.linear_model import Ridge  # RidgeClassier is also there

li_reg = LinearRegression()
lm_lasso = Lasso() 
lm_ridge =  Ridge() 

li_reg.fit(features_train, labels_train)
lm_lasso.fit(features_train, labels_train)
lm_ridge.fit(features_train, labels_train)


print ("RSquare Value for Simple Regresssion TEST data is-") 
print (np.round (li_reg.score(features_test,labels_test)*100,2)) # 26.43%

print ("RSquare Value for Lasso Regresssion TEST data is-")
print (np.round (lm_lasso.score(features_test,labels_test)*100,2)) # -64.02%

print ("RSquare Value for Ridge Regresssion TEST data is-")
print (np.round (lm_ridge.score(features_test,labels_test)*100,2)) # 26.81%

predict_test_linear = li_reg.predict(features_test)
predict_test_lasso = lm_lasso.predict (features_test) 
predict_test_ridge = lm_ridge.predict (features_test)

print ("Simple Regression Mean Square Error (MSE) for TEST data is") # 0.1
print (np.round (metrics.mean_squared_error(labels_test, predict_test_linear),2))

print ("Lasso Regression Mean Square Error (MSE) for TEST data is") # 0.22
print (np.round (metrics .mean_squared_error(labels_test, predict_test_lasso),2))

print ("Ridge Regression Mean Square Error (MSE) for TEST data is") # 0.1
print (np.round (metrics .mean_squared_error(labels_test, predict_test_ridge),2))

print('By applying above algorithm, we can say that ridge regression is better than other.')





