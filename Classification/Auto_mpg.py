"""
Q2. (Create a program that fulfills the following specification.)

Here is the dataset about cars. The data concerns city-cycle fuel consumption in 
miles per gallon (MPG).

Import the dataset Auto_mpg.txt
Give the column names as "mpg", "cylinders", "displacement","horsepower","weight",
"acceleration", "model year", "origin", "car name" respectively

1.Display the Car Name with highest miles per gallon value
2.Build the Decision Tree and Random Forest models and find out which of the two is 
  more accurate in predicting the MPG value

Find out the MPG value of a 80's model car of origin 3, weighing 2630 kgs with 6 
cylinders, having acceleration around 22.2 m/s due to it's 100 horsepower engine 
giving it a displacement of about 215. (Give the prediction from both the models)
"""

import pandas as pd
import numpy as np

"""
Data Processing
"""
## opening the file and making the DataFrame to analyze
f= open('Auto_mpg.txt','r')   
content= [ line.strip().split() for line in f ]

for i in range( len(content)):
    y= content[i][:8]
    y.append(' '.join(content[i][8:]).strip('"') )
    content[i]= y

print(content)

data= pd.DataFrame(content, columns=["mpg", "cylinders", "displacement","horsepower","weight", "acceleration", "model year", "origin", "car name"])
f.close()
## numeric columns but with str dtype
column= ["mpg", "cylinders", "displacement","horsepower","weight", "acceleration", "model year", "origin"]

## handle missing values
data["horsepower"]= data["horsepower"].apply(lambda x: x.replace('?','90'))

## converting object dtype to float
data[column] = data[column].apply(np.float64)
data.info()

####################################

'''
Display the Car Name with highest miles per gallon value
'''
car = data[data["mpg"]== max(data["mpg"])]["car name"]
print(car)

##################################
'''
Build the Decision Tree
'''

# Checking for Categorical Data
data.head()
pd.set_option('display.max_columns', None)
data.info()

# Finding missing data
data.isnull().any(axis=0)

features = data.drop(['mpg','car name'], axis=1)  
labels = data['mpg'] 

from sklearn.model_selection import train_test_split  
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=0) 

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()  
features_train = sc.fit_transform(features_train)  
features_test = sc.transform(features_test)  

from sklearn.tree import DecisionTreeRegressor  
regressor = DecisionTreeRegressor()  
regressor.fit(features_train, labels_train)  

labels_pred = regressor.predict(features_test)

df=pd.DataFrame({'Actual':labels_test, 'Predicted':labels_pred})  
print(df)  

#Evaluating the algorithm
from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(labels_test, labels_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(labels_test, labels_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(labels_test, labels_pred)))  

print(regressor.score(features_train, labels_train)*100)  ## 100%
print(regressor.score(features_test, labels_test)*100)   ## 88.34%
print(regressor.score(features_test, labels_pred)*100)  ##100%

####################################################

'''
Random forest decision tree
'''

#train the model
from sklearn.ensemble import RandomForestRegressor
regressor2 = RandomForestRegressor(n_estimators=200, random_state=0)  
regressor2.fit(features_train, labels_train)  

labels_pred = regressor2.predict(features_test)

df=pd.DataFrame({'Actual':labels_test, 'Predicted':labels_pred})  
print(df) 

#Evaluating the algorithm
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(labels_test, labels_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(labels_test, labels_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(labels_test, labels_pred)))  
#print (np.mean(labels))

print(regressor2.score(features_train, labels_train)*100)  ## 98.20%
print(regressor2.score(features_test, labels_test)*100)   ## 89.54%
print(regressor2.score(features_test, labels_pred)*100)  ##100%


print('''By performing both algorithm, we can conclude that Random Forest Regression is 
slightly better than Decision Tree Regression.''')

####################################################
"""
Find out the MPG value of a 80's model car of origin 3, weighing 2630 kgs with 6 
cylinders, having acceleration around 22.2 m/s due to it's 100 horsepower engine 
giving it a displacement of about 215. (Give the prediction from both the models)
"""

x= np.array([6, 215, 100, 2630, 22.2, 80, 3])
pred = regressor.predict([x])
print("MPG value for given car:",pred[0]) ## 15.5

## by random forest
pred = regressor2.predict([x])
print("MPG value for given car:",pred[0]) ## 17.10




