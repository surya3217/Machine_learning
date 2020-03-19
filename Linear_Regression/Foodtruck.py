"""
Code Challenge
  Problem Statement:
    Suppose you are the CEO of a restaurant franchise and are considering 
    different cities for opening a new outlet. 
    
    The chain already has food-trucks in various cities and you have data for profits 
    and populations from the cities. 
    
    You would like to use this data to help you select which city to expand to next.
    
    Perform Simple Linear regression to predict the profit based on the 
    population observed and visualize the result.
    
    Based on the above trained results, what will be your estimated profit, 
    
    If you set up your outlet in Jaipur? 
    (Current population in Jaipur is 3.073 million)
  Hint: 
    You will implement linear regression to predict the profits for a 
    food chain company.
    Foodtruck.csv contains the dataset for our linear regression problem. 
    The first column is the population of a city and the second column is the 
    profit of a food truck in that city. 
    A negative value for profit indicates a loss.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('All CSV/Foodtruck.csv')

features= dataset.iloc[:,:-1].values  ## return 2D array
labels= dataset.iloc[:,-1].values


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.2,random_state = 0)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()              ## generate linear function
regressor.fit(features_train, labels_train)   ## take only 2D array 



# Predicting the Test set results
labels_pred = regressor.predict(features_test)  
print (pd.DataFrame({'Actual': labels_test, 'Predicted': labels_pred} ))


print(regressor.score(features_test, labels_test)*100)  ## test score
print(regressor.score(features_train, labels_train)*100) ## train score


# Visualising the Training set results
plt.scatter(features_train, labels_train, color = 'red')
plt.plot(features_train, regressor.predict(features_train), color = 'blue')

x= np.array([3.073])  ## jaipur population
x= x.reshape(1,1)
y= regressor.predict(x)

if y<0 :
    print('No profit at this location')
else:
    print('Profit',y)



