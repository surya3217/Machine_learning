"""
Q2. (Create a program that fulfills the following specification.)
Female_Stats.Csv

Female Stat Students

Import The Female_Stats.Csv File
The Data Are From N = 214 Females In Statistics Classes At The University Of California At Davi.

Column1 = Student’s Self-Reported Height,
Column2 = Student’s Guess At Her Mother’s Height, And
Column3 = Student’s Guess At Her Father’s Height. All Heights Are In Inches.

1. Build A Predictive Model And Conclude If Both Predictors (Independent Variables) Are Significant For A Students’ Height Or Not.
2. When Father’s Height Is Held Constant, The Average Student Height Increases By How Many Inches For Each One-Inch Increase In Mother’s Height.
3. When Mother’s Height Is Held Constant, The Average Student Height Increases By How Many Inches For Each One-Inch Increase In Father’s Height.

"""

"""
1. Build A Predictive Model And Conclude If Both Predictors (Independent Variables) Are 
   Significant For A Students’ Height Or Not.
"""

import pandas as pd
import numpy as np

# Importing the dataset
data = pd.read_csv('All CSV/Female_Stats.Csv')
data.info()

# Check for missing data column wise
data.isnull().any(axis=0)

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
regressor.fit(features_train, labels_train)

# y = ax1 + bx2 + c
# Here a, b and c are the coefficients and d is the intercept
print(regressor.intercept_)   
print (regressor.coef_)
print ('Training Score:',regressor.score(features_train, labels_train)*100) # 43.36%


# Predicting the Test set results
Pred = regressor.predict(features_test)
print (pd.DataFrame(zip(Pred, labels_test), columns= ['Predicted', 'Actual'] ))
print ('Testing Score:',regressor.score(features_test, Pred)*100)  # 100% 

#######################################################################


"""
2. When Father’s Height Is Held Constant, The Average Student Height Increases By How 
   Many Inches For Each One-Inch Increase In Mother’s Height.
"""

for i in range(60,70):
    x= [[i, 65]]
    print(regressor.predict(x) )    

print("""Here we can conclude that after incrementing the height of mother by every 
'4 inches', average student's height incremented by '1 inches'.""")


#######################################
    
"""
3. When Mother’s Height Is Held Constant, The Average Student Height Increases By How 
   Many Inches For Each One-Inch Increase In Father’s Height.
"""

for i in range(60,71):
    x= [[65, i]]
    print(regressor.predict(x) )    

print("""Here we can conclude that after incrementing the height of father by every 
'2 inches', average student's height incremented by '1 inches'.""")

########################################

"""
Checking the significance of both predictors using OLS method.
"""
## stats model module use with numpy module to delete

import statsmodels.api as sm

features_obj = features[:, [0,1]]
features_obj = sm.add_constant(features_obj)
while (True):
    regressor_OLS = sm.OLS(endog = labels,exog =features_obj).fit()
    p_values = regressor_OLS.pvalues
    if p_values.max() > 0.05 :
        features_obj = np.delete(features_obj, p_values.argmax(),1)
    else:
        break

regressor_OLS.summary()

print("""After using OLS method, we can conlude that both predictors are significant
to mesure the student's height.""")



