
"""
Code Challenges

Q1. (Create a program that fulfills the following specification.)
Import the affairs.csv file.

It was derived from a survey of women in 1974 by Redbook magazine, in which married women 
were asked about their participation in extramarital affairs.

Description of Variables:

The dataset contains 6366 observations of 10 variables(modified and cleaned):

1.rate_marriage: woman's rating of her marriage (1 = very poor, 5 = very good)
2.age: women's age
3.yrs_married: number of years married
4.children: number of children
5.religious: women's rating of how religious she is (1 = not religious, 4 = strongly religious)
6.educ: level of education (9 = grade school, 12 = high school, 14 = some college, 16 = college graduate, 17 = some graduate school, 20 = advanced degree)
7.occupation: women's occupation (1 = student, 2 = farming/semi-skilled/unskilled, 3 = "white collar", 4 = teacher/nurse/writer/technician/skilled, 5 = managerial/business, 6 = professional with advanced degree)
8.occupation_husb: husband's occupation (same coding as above)
9.affair: outcome 0/1, where 1 means a woman had at least 1 affair.

Task:
Now, perform Classification using logistic regression and check your
model accuracy using confusion matrix and also through .score() function.

NOTE: 
1. Perform OneHotEncoding for occupation and occupation_husb, since they should be 
   treated as categorical variables. Careful from dummy variable trap for both!!

2. What percentage of total women actually had an affair?
(note that Increases in marriage rating and religiousness correspond to a decrease in the likelihood of having an affair.)

3. Predict the probability of an affair for a random woman not present in the dataset.
   She's a 25-year-old teacher who graduated college, has been married for 3 years, 
   has 1 child, rates herself as strongly religious, rates her marriage as fair, and 
   her husband is a farmer.
 x= [0,0,1,0,0, 1,0,0,0,0 , 3, 25, 3,1, 4,16]
 
Optional:
    Build an optimum model, observe all the coefficients.
"""


import sklearn as sk
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('All CSV/affairs.csv', sep=',',header=0)  
data.info()
data.sample(10)

features= data.iloc[:, :-1]
labels= data.iloc[:, -1]

"""
1. Perform OneHotEncoding for occupation and occupation_husb, since they should be 
   treated as categorical variables. Careful from dummy variable trap for both!!
"""
from sklearn.preprocessing import OneHotEncoder

## column 7 for ouccuption_husb 
# Creation of Object
onehotencoder = OneHotEncoder(categorical_features = [7])  ## making a list for the column which have object data 

# Convert to NDArray format
features = onehotencoder.fit_transform(features).toarray()
# OneHotEncoder always puts the encoded values as the first columns
# irrespective of the column you are encoding

print(features)

# Avoiding the Dummy Variable Trap
features = features[:, 1:]


## column 12 for ouccuption women
# Creation of Object
onehotencoder = OneHotEncoder(categorical_features = [11])  ## making a list for the column which have object data 
features = onehotencoder.fit_transform(features).toarray()
print(features)

# Avoiding the Dummy Variable Trap
features = features[:, 1:]



# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
features_scale = sc.fit_transform(features)


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features_scale, labels, test_size = 0.2, random_state = 0)


## Feature Scaling
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#features_train = sc.fit_transform(features_train)

# He has already calculated the mean and sd, so we only need to transform
#features_test = sc.transform(features_test)



# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(features_train, labels_train)


#Calculate Class Probabilities
# Open in variable explorer and show ( FAIL and PASS probability)
# There are two probabilities since we have 2 class
probability = classifier.predict_proba(features_test)
print(probability)

## score
print( classifier.score(features_train, labels_train)*100 ) ## 72.15 %


# Predicting the class labels ( 0 or 1 )
labels_pred = classifier.predict(features_test)

# Comparing the predicted and actual values
my_frame= pd.DataFrame(labels_pred, labels_test)
print(my_frame)

print( classifier.score(features_test, labels_test)*100 ) ## 74.33 %
#print( classifier.score(features_test, classifier.predict(features_test) )*100 )


"""
2. What percentage of total women actually had an affair?
"""
# Now we can draw 4 combination 
# I predicted 1 , Acutal was also 1  ( RIGHT PREDICTION ) True positives (TP)
# I predicted 2,  Actual was also 2  ( RIGHT PREDICTION ) True negatives (TN)

# I predicted 1 , Acutal was also 2  ( WRONG PREDICTION ) False positives (FP)
# I predicted 2,  Actual was also 1  ( WRONG PREDICTION ) False negatives (FN)

# Making the Confusion Matrix or Error Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels_test, labels_pred)
print(cm)

"""
    1   2   ( predicted)
1 [804  81]
2 [246 143]

804 + 143  ( RIGHT PREDICTION )
81 + 246   ( WRONG PREDICTION )
"""

# Model Score = 74.33 times out of 100 model prediction was RIGHT in testing
print('Score: ', ((cm[0][0] + cm[1][1]) / sum(sum(cm)))*100 )  ## 74.33 %


labels_pred2 = classifier.predict(features_scale)
cm = confusion_matrix(labels, labels_pred2)
print(cm)

"""
[[3885  428]
 [1317  736]]
"""
perc= ((cm[0][0] + cm[1][1]) / sum(sum(cm)))*100
print('Total percentage affairs: ',  perc)  ## 72.58 %

######################################################

"""
3. Predict the probability of an affair for a random woman not present in the dataset.
   She's a 25-year-old teacher who graduated college, has been married for 3 years, 
   has 1 child, rates herself as strongly religious, rates her marriage as fair, and 
   her husband is a farmer.
"""

x= np.array([[0,0,1,0,0, 1,0,0,0,0 , 3, 25, 3,1, 4,16]])  ## input
x= sc.fit_transform(x)

probability = classifier.predict_proba(x)
print(100*probability)

########################################################

"""
    Build an optimum model, observe all the coefficients.
"""

# code to automate the p value removing
import statsmodels.api as sm
import numpy as np

features_obj = features_scale[:, :]
features_obj = sm.add_constant(features_obj)
temp= features_obj

while (True):
    regressor_OLS = sm.OLS(endog = labels,exog =features_obj).fit()
    p_values = regressor_OLS.pvalues
    if p_values.max() > 0.05 :
        features_obj = np.delete(features_obj, p_values.argmax(), 1)
    else:
        break

print(features_obj)   
regressor_OLS.summary()

##   0,1,2,3,4,  5,6,7,8,9,  10,11,12,13,14, 15  column number
##  [0,0,1,0,0,  1,0,0,0,0,  3, 25, 3, 1, 4, 16] input for women

## Only these columns are significance in computation [ 0, 2, 4, 5, 11, 12, 13, 15 ]

"""
0,2,4: women occupation
5:  occupation_husb
11: age
12: yrs_maeeied
13: children
15: educ
"""


###################################################################################

# Another method

import numpy as np
import pandas as pd

# Reading data from csv
dataset = pd.read_csv("affairs.csv")

# Separating data into Independent and Dependent Variables
fe = dataset.iloc[:,:-1].values
la = dataset.iloc[:,-1].values

def Model(features, labels):
    # Applying OneHotEncoding
    from sklearn.preprocessing import OneHotEncoder
    
    col_to_ohe = [6,7]  # Columns to be OneHotEncoded
    ohe=OneHotEncoder(categorical_features=[col_to_ohe])
    features = ohe.fit_transform(features).toarray()
    
    # Getting indexes for the columns to be dropped, to avoid dummy variable trap
    total_col, indexes = 0, []
    for col in col_to_ohe:
        unique_val_count = len(dataset.iloc[:,col].value_counts())
        total_col += unique_val_count
        indexes.append(total_col - unique_val_count)
    
    # Dropping the dummy variable trap columns
    features = np.delete(features, indexes, axis=1)
    
    # Splitting the dataset into train and test
    from sklearn.model_selection import train_test_split as TTS
    
    f_train,f_test,l_train,l_test = TTS(features, labels, test_size = 0.25,
                                        random_state = 0)
    
    # Logistic Regression Model
    from sklearn.linear_model import LogisticRegression
    reg = LogisticRegression(random_state=0)
    reg = reg.fit(f_train, l_train)
    
    pred = reg.predict(f_test)   # Prediction on test data
    
    # np.array([1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 3, 25, 3, 1, 4, 16]).reshape(1,-1)
    # Preprocessing the new individual's data
    val = np.array([3, 25, 3, 1, 4, 16, 4, 2]).reshape(1,-1)
    val = ohe.transform(val).toarray()
    val = np.delete(val, indexes, axis=1)
    
    val_pred = reg.predict_proba(val)  # Predicting Individual's value
    
    # Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(l_test, pred)
    
    # check the accuracy on the Model
    mod_score = reg.score(f_test, l_test)
    
    return pred,val_pred,cm,mod_score

Pred, val_Pred, CM, Score = Model(fe,la)

print ("model accuracy using confusion matrix : "+str(CM))
print ("model accuracy using .score() function : "+str(round(Score*100,2)))
print ("percentage of total women actually had an affair : "+str(round(dataset["affair"].mean()*100,2))+"%")
print ("probability of an affair for a random woman is : "+str(val_Pred)) 






