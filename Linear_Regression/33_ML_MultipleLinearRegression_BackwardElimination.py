
# Multiple Linear Regression

# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('All CSV/Salary_Classification.csv')
#temp = dataset.values
features = dataset.iloc[:, :-1].values
labels = dataset.iloc[:, -1].values

# Encoding categorical data
"""
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
features[:, 0] = labelencoder.fit_transform(features[:, 0])
"""
#https://towardsdatascience.com/columntransformer-in-scikit-for-labelencoding-and-onehotencoding-in-machine-learning-c6255952731b
#from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
features = np.array(columnTransformer.fit_transform(features), dtype = np.float32)


#x = preprocess.fit_transform(features)
features = features[:,1:]   ## removing redundant column


'''or'''
"""
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [0])
features = onehotencoder.fit_transform(features).toarray()
"""

# Avoiding the Dummy Variable Trap
# dropping first column
#features = features[:, 1:]


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.2, random_state = 0)


# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(features_train, labels_train)


# Predicting the Test set results
Pred = regressor.predict(features_test)

import pandas as pd
import numpy as np
print (pd.DataFrame(zip(Pred, labels_test)))


# Prediction for a new values 
# make this accorifng to the data csv
# Development is replaced by 1,0,0 to 0,0 to remove dummy trap

x = ['Development',1150,3,4]

x = np.array(x)
x = x.reshape(1,-1)  ## convrting 1d to 2d array
## tranforming object to float value
x = np.array(columnTransformer.transform(x), dtype = np.float32)

x = x[:,1:]
regressor.predict(x)

'''or'''
"""
le = labelencoder.transform(['Development'])
ohe = onehotencoder.transform(le.reshape(1,1)).toarray()
x = [ohe[0][1],ohe[0][2],1150,3,4]
x = np.array(x)
regressor.predict(x.reshape(1, -1))
"""

# Getting Score for the Multi Linear Reg model
Score1 = regressor.score(features_train, labels_train)
Score2 = regressor.score(features_test, labels_test)

# to print the values of weights or coefficients
print (regressor.coef_)







## finding the best fit column which effecting the prediction majorly

# Building the optimal model using Backward Elimination
#import statsmodels.formula.api as sm
import statsmodels.regression.linear_model as sm

#This is done because statsmodels library requires it to be done for constants.
features = np.append(arr = np.ones((30, 1)), values = features, axis = 1)

## axis 0 means row wise addition 
## axis 1 means column wise addition 

features_opt = features[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = labels, exog = features_opt).fit()
regressor_OLS.summary()

## checking the p value
## if p value is 'maximum' then its importance is 'very low'
## if p value is 'minimum' then its importance is 'very high'
## if p value is 5% of the maximum then no column will remove

features_opt = features[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = labels, exog = features_opt).fit()
regressor_OLS.summary()




features_opt = features[:, [0, 1, 3, 5]]
regressor_OLS = sm.OLS(endog = labels, exog = features_opt).fit()
regressor_OLS.summary()



features_opt = features[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog = labels, exog = features_opt).fit()
regressor_OLS.summary()



features_opt = features[:, [0, 5]]
regressor_OLS = sm.OLS(endog = labels, exog = features_opt).fit()
print (regressor_OLS.summary())

regressor_OLS.pvalues


'''Or'''

features_opt = features[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = labels, exog = features_opt).fit()
regressor_OLS.summary()
a= regressor_OLS.pvalues


#a= [1.21990393e-04, 6.01172763e-01, 9.73199331e-01, 1.78972087e-01, 7.43907416e-01, 1.08255925e-15]
column= [0, 1, 2, 3, 4, 5]
a= regressor_OLS.pvalues



list_= list(range(len(features[0])))
while(True):
    regressor_OLS = sm.OLS(endog = labels, exog = features_opt).fit()

    if regressor_OLS.pvalues.max >=0.5:
        del list_[regressor_OLS.pvalues.argmax()]
        print(list_)
        features_opt = features[:, list_]
    else:
        break
    
    
    

while(max(a)> 0.05):
    
    features_opt = features[:, column]
    regressor_OLS = sm.OLS(endog = labels, exog = features_opt).fit()
    regressor_OLS.summary()
#    a= regressor_OLS.pvalues

    d= list(regressor_OLS.pvalues)
    column.pop(d.index(max(d)) )
    print(regressor_OLS.pvalues)
    print(column)
     


"""
Few comments about OLS for dummy variable values

Case Study
Suppose you are building a linear (or logistic) regression 
model. In your independent variables list, you have a 
categorical variable with 4 categories (or levels). 
You created 3 dummy variables (k-1 categories) and 
set one of the category as a reference category. 
Then you run stepwise / backward/ forward regression 
technique and you found only one of the category coming 
out statistically significant based on p-value and the 
remaining 3 categories are insignificant. 
The question arises - should we remove or keep these 3 
categories having insignificant difference? should we 
include the whole categorical variable or not?

Solution
In short, the answer is we can ONLY choose whether we 
should use this independent categorical variable as a 
whole or not. In other words, we should only see whether 
the categorical variable as a whole is significant or not. 
We cannot include some categories of a variable and exclude 
some categories having insignificant difference.

Ref: https://www.listendata.com/2016/07/insignificant-levels-of-categorical-variable.html
"""



"""
//Dataset and business problem

1. We have understood the simple linear regression and how to prepare a model on that.

The example we used was based on years of experience and salary.

That was simple regressor case.

2. Now lets understand multiple regression. For this, we take some data about startups in US.



This startup data tells about companies (how much they spend on R&D, how much they spend on administration and marketing and what is their location).

On above parameters, we need to figure which companies are performing better.
This sort of data can be well utilized with venture capitalist as want to to invest in high performing startups.

3. So their goal is maximizing profits.

5. The data is about 50 startups whose names are not given. but other parameters are well given.

------------------------------------------------
The above problem is about multiple regression

------------------------------------------
What is multiple regression?
//same as simple regression but multiple independent variables

//Assumptions of linear regression - do not blindly follow but check if the given problem is linear regression problem

// In above problem, profit is our dependent variables and others are independent variables

//but state (location) is a categorical variable and we need to handle this.
//For this you need to create dummy variables
//Dummy variables trap ( include one less dummy variables in predictor)
// Whenever you are building a model, always omit one dummy variables.
// So beware of dummy variable trap
//Show the image (dummy variable trap)
// Now lets start buidling a the model

// Some times you have to miss some of independent variable for model buidling (elimination) [selecting the right variables)
// All possible methods are given below
1. All in
2. Backward elimination
3. Forward selection
4. Bidirection elimination
5. Score comparison

 
----------------
In most of the cases library takes care of dummy variable trap and feature scaling as well.
//Explain the sample code

// Compare the prediction with actual data points.

// Now how you can improve the model?
//By checking which independent variables has highest impact?
// For this we use method  called backward elimination
// Explain the backward elimination
//Steps
1. Start with all the predictors in the model
2. Remove the predictor with highest p-value greater than 5%
3. Refit the model and goto 2
4. Stop when all p-values are less than 5%.

// Need to use library for this (import stats_models.formula.api as sm)

//Using this libarary, we remove predictors(independent variables) iteratively by looking at p value (remove if p is more than 5%)


-------------------------------------------------------------------------------
--------------------------------------------------------------------------------
<Polynomial Linear Regression>


-------------------------------------
Other non linear regression
//SVR
//Most of libraries has inbuilt feature scaling but SVR class does not apply feature scaling



// Most important

https://www.analyticsvidhya.com/blog/2016/01/complete-tutorial-ridge-lasso-regression-python/




Q1. (Create a program that fulfills the following specification.)
iq_size.csv

Are a person's brain size and body size (Height and weight) predictive of his or her intelligence?

 

Import the iq_size.csv file

It Contains the details of 38 students, where

Column 1: The intelligence (PIQ) of students

Column 2:  The brain size (MRI) of students (given as count/10,000).

Column 3: The height (Height) of students (inches)

Column 4: The weight (Weight) of student (pounds)

    What is the IQ of an individual with a given brain size of 90, height of 70 inches, and weight 150 pounds ? 
    Build an optimal model and conclude which is more useful in predicting intelligence Height, Weight or brain size.




Q2. (Create a program that fulfills the following specification.)
Female_Stats.Csv

Female Stat Students

 

Import The Female_Stats.Csv File

The Data Are From N = 214 Females In Statistics Classes At The University Of California At Davi.

Column1 = Student’s Self-Reported Height,

Column2 = Student’s Guess At Her Mother’s Height, And

Column 3 = Student’s Guess At Her Father’s Height. All Heights Are In Inches.

 

    Build A Predictive Model And Conclude If Both Predictors (Independent Variables) Are Significant For A Students’ Height Or Not
    When Father’s Height Is Held Constant, The Average Student Height Increases By How Many Inches For Each One-Inch Increase In Mother’s Height.
    When Mother’s Height Is Held Constant, The Average Student Height Increases By How Many Inches For Each One-Inch Increase In Father’s Height.




