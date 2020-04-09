"""
Name:
  University Admission Prediction Tool
File Name:
  uni_admin.py
Dataset:
  University_data.csv
Problem Statement:
  Perform Linear regression to predict the chance of admission based on all the features given.
  Based on the above trained results, what will be your estimated chance of admission.
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset (Bivariate Data Set with 3 Clusters)
dataset = pd.read_csv('All CSV/University_data.csv')
dataset.info()

features = dataset.iloc[:, 1:-1].values
labels = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
                  train_test_split(features, labels, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
features_train = sc.fit_transform(features_train)
features_test = sc.transform(features_test)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(features_train, labels_train)

# Check the value of intercept and slope
# y = ax + by + cz + d
# Here a, b and c are the coefficients and d is the intercept
print(regressor.intercept_)   
print (regressor.coef_)   

print ("Score for Linear Regresssion TEST data is-") 
print (np.round (regressor.score(features_test,labels_test)*100,2)) # 76.08 %

# Predicting the Test set results
labels_pred = regressor.predict(features_test)
print (pd.DataFrame(zip(labels_pred, labels_test), columns= ['Predicted', 'Actual'] ))

# Evaluate the Algorithm 
from sklearn import metrics  
print('Root Mean Squared Error:',np.sqrt(metrics.mean_squared_error(labels_test, labels_pred))) 
print('Mean value:', np.mean(labels))
print('As RMS < 10 % of Mean value, so this algorithm is good for evaluation.')
####################################################################################

"""
Checking the significance of both predictors using OLS method.
"""
## stats model module use with numpy module to delete

import statsmodels.api as sm
features = sc.transform(features)

features_obj = features[:, [0,1,2,3,4]]
features_obj = sm.add_constant(features_obj)
while (True):
    regressor_OLS = sm.OLS(endog = labels,exog =features_obj).fit()
    p_values = regressor_OLS.pvalues
    if p_values.max() > 0.05 :
        features_obj = np.delete(features_obj, p_values.argmax(),1)
    else:
        break

regressor_OLS.summary()
print("""After using OLS method, we can conlude that  SOP column is less significant
in evaluation.""")



