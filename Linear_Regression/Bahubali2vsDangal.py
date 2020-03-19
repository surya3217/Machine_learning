
"""
Code Challenge
  Problem Statement:
    It contains Data of Day wise collections of the movies Bahubali 2 and Dangal 
    (in crores) for the first 9 days.
    
    Now, you have to write a python code to predict which movie would collect 
    more on the 10th day.
  Hint:
    First Approach - Create two models, one for Bahubali and another for Dangal
    Second Approach - Create one model with two labels
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data= pd.read_csv('All CSV/Bahubali2_vs_Dangal.csv')

# Check Column wise is any data is missing or NaN
data.isnull().any(axis=0)

# Check data Types for each columns
print(data.dtypes)

features= data.iloc[:,:-2].values  ## day
Blabels= data.iloc[:,-2].values
Dlabels= data.iloc[:,2:].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
features_train, features_test, Blabels_train, Blabels_test, Dlabels_train, Dlabels_test = train_test_split(features, Blabels, Dlabels, test_size = 0.2,random_state = 0)

## for BAHUBALI movie
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()          ## generate linear function
regressor.fit(features_train, Blabels_train)    ## take only 2D array 
Blabels_pred = regressor.predict(features_test)  
print( pd.DataFrame( {'Actual': Blabels_test, 'Predicted': Blabels_pred} ) )


# Visualising the Training set results
plt.scatter(features_train, Blabels_train, color = 'red')
plt.plot(features_train, regressor.predict(features_train), color = 'blue')
plt.title('Days vs Bahubali2_Collection (Training set)')
plt.xlabel('Days')
plt.ylabel('Bahubali2_Collection')
plt.show()



## for DANGAL movie
regressor2 = LinearRegression() 
regressor2.fit(features_train, Dlabels_train)
Dlabels_pred = regressor2.predict(features_test)  

Dlabels_test= Dlabels_test.flatten()  ## convert 2d arrray to 1d  array
Dlabels_pred= Dlabels_pred.flatten()
print( pd.DataFrame({ 'Actual': Dlabels_test, 'Predicted': Dlabels_pred } ) )

# Visualising the Training set results
plt.scatter(features_train, Dlabels_train, color = 'red')
plt.plot(features_train, regressor.predict(features_train), color = 'blue')
plt.title('Days vs Dangal_Collection (Training set)')
plt.xlabel('Days')
plt.ylabel('Dangal_Collection')
plt.show()

#Model accuracy for BAHUBALI
print ('Function accuracy for BAHUBALI: ',regressor.score(features_train, Blabels_train)*100,'%')
print ('Testing: ',regressor.score(features_test, Blabels_test)*100,'%')


#Model accuracy for DANGAL
print ('Function accuracy for DANGAL: ',regressor2.score(features_train, Dlabels_train)*100,'%')
print ('Testing: ',regressor2.score(features_test, Dlabels_test)*100,'%')


## prediction for the movies on 10th day
x= np.array([10])
x= x.reshape(1,1)
print( 'BAHUBALI movie prediction on 10th day: ',regressor.predict(x)) ## BAHUBALI movie pred.
print('DANGAL movie prediction on 10th day: ',regressor2.predict(x).flatten() ) ## DANGAL movie pred.





