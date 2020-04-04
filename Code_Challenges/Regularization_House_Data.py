"""
Code Challenges 02: (House Data) kc_house_data.csv

This is kings house society data.
In particular, we will: 
• Use Linear Regression and see the results
• Use Lasso (L1) and see the resuls
• Use Ridge and see the score
"""

#Import libraries
import pandas as pd  
import numpy as np  

#import database
dataset = pd.read_csv('All CSV/kc_house_data.csv') 
dataset.info()
dataset.head()

# Finding missing data
dataset.isnull().any(axis=0)

# Handle missing values
dataset['sqft_above']= dataset['sqft_above'].fillna(np.mean(dataset['sqft_above']))
dataset.info()

features = dataset.iloc[:, 3:].values  
labels = dataset.iloc[:, 2].values

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


print ("Simple Regresssion TEST data is-") 
print (np.round (li_reg.score(features_test,labels_test)*100,2)) # 69.01%

print ("Lasso Regresssion TEST data is-")
print (np.round (lm_lasso.score(features_test,labels_test)*100,2)) # 69.01%

print ("Ridge Regresssion TEST data is-")
print (np.round (lm_ridge.score(features_test,labels_test)*100,2)) # 69.01%

predict_test_linear = li_reg.predict(features_test)
predict_test_lasso = lm_lasso.predict (features_test) 
predict_test_ridge = lm_ridge.predict (features_test)

from sklearn import metrics 

print ("Simple Regression Root Mean Square Error (MSE) for TEST data is") #202899.39
print (np.sqrt (metrics.mean_squared_error(labels_test, predict_test_linear)))

print ("Lasso Regression Root Mean Square Error (MSE) for TEST data is") #202899.57
print (np.sqrt (metrics .mean_squared_error(labels_test, predict_test_lasso)))

print ("Ridge Regression Root Mean Square Error (MSE) for TEST data is") #202900.00
print (np.sqrt (metrics .mean_squared_error(labels_test, predict_test_ridge)))

print('''By applying above algorithm, we can say that Simple linear regression and 
Lasso Regression, both performance is almost same and better than other Ridge 
Regression.''')





