"""
Q1. (Create a program that fulfills the following specification.)
iq_size.csv

Are a person's brain size and body size (Height and weight) predictive of his or her intelligence?
Import the iq_size.csv file
It Contains the details of 38 students, where

Column 1: The intelligence (PIQ) of students
Column 2: The brain size (MRI) of students (given as count/10,000).
Column 3: The height (Height) of students (inches)
Column 4: The weight (Weight) of student (pounds)

1. What is the IQ of an individual with a given brain size of 90, height of 70 inches, and weight 150 pounds ? 
2. Build an optimal model and conclude which is more useful in predicting intelligence Height, Weight or brain size.
"""

import pandas as pd
import numpy as np

# Importing the dataset
data = pd.read_csv('All CSV/iq_size.csv')
data.info()

features= data.iloc[:, 1:].values
labels= data.iloc[:,0].values

# Dataset is small so no splitting
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test= train_test_split(features, labels, test_size= 0.2, random_state= 1)


# Fitting Multiple Linear Regression to the Training set
# Whether we have Univariate or Multivariate, class is LinearRegression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(features, labels)


# y = ax + by + cz + d
# Here a, b and c are the coefficients and d is the intercept
print(regressor.intercept_)   
print (regressor.coef_)
print (regressor.score(features, labels)*100)   # 29.49 %

# We cannot show a line on a graph as we did for 2D data, since we have 5D data
# Predicting the Test set results
Pred = regressor.predict(features_test)
print (pd.DataFrame(zip(Pred, labels_test), columns= ['Predicted', 'Actual'] ))
print (regressor.score(features_test, labels_test)*100)


x= np.array([90,70,150])
x= x.reshape(1,3)
out = regressor.predict(x)
print('Iq size of student:',out)

#################################

"""
2. Build an optimal model and conclude which is more useful in predicting intelligence 
   Height, Weight or brain size.
"""

# code to automate the p value removing
import statsmodels.api as sm
import numpy as np

features_obj = features[:, [0,1,2]]
features_obj = sm.add_constant(features_obj)

while (True):
    regressor_OLS = sm.OLS(endog = labels,exog =features_obj).fit()
    p_values = regressor_OLS.pvalues
    if p_values.max() > 0.05 :
        features_obj = np.delete(features_obj, p_values.argmax(),1)
    else:
        break

print(features_obj)   ## Brain size 
regressor_OLS.summary()

print('From the OLS method we conclude that "Brain size" is most significant in predicting intelligence.')







