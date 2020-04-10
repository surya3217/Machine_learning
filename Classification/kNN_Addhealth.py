"""
Q2. (Create a program that fulfills the following specification.)
Import tree_addhealth.csv

For this Code Challenge, The National Longitudinal Study of Adolescent to Adult Health
(Add Health) data set, an ongoing (longitudinal) survey study that began in the mid-1990s
is used. The project website URL is:

This large data set is available online from the University of North Carolina’s 
Carolina Population Center, http://www.cpc.unc.edu/projects/addhealth/data.

The attributes are:
1.BIO_SEX   : 1 = male 0 = female    
2.HISPANIC  : 1=Yes,0=No    
3.WHITE     : 1=Yes,0=No
4.BLACK     : 1=Yes,0=No          
5.NAMERICAN : 1=Yes,0=No                      
6.ASIAN     : 1=Yes,0=No                      
7.ALCEVR1   : ever drank alcohol(1=Yes,0=No)
8.marever1  : ever smoked marijuana(1=Yes,0=No)    
9.cocever1  : ever used cocaine(1=Yes,0=No)                
10.inhever1 : ever used inhalants(1=Yes,0=No)             
11.cigavail : cigarettes available in home(1=Yes,0=No)
12.PASSIST  : parents or public assistance(1=Yes,0=No)
13.EXPEL1   : ever expelled from school(1=Yes,0=No)
14.TREG1    : Ever smoked regularly(1=Yes,0=No)

#Explanatory Variables:
1.Age
2.ALCPROBS1: alcohol problems 0-6
3.DEP1     : depression scale
4.ESTEEM1  : self esteem scale       
5.VIOL1    : violent behaviour scale
6.DEVIANT1 : deviant behaviour scale     
7.SCHCONN1 : school connectedness scale       
8.GPA1     : gpa scale  4 points)
9.FAMCONCT : family connectedness scale       
10.PARACTV : parent activities scale
11.PARPRES : parental presence scale

(Please make confusion matrix and also check accuracy score for each and every 
section)
"""
"""
  1.Build a classification tree model evaluating if an adolescent would smoke 
    regularly or not based on: gender, age, (race/ethnicity) Hispanic, White, Black, 
    Native American and Asian, alcohol use, alcohol problems, marijuana use, cocaine 
    use, inhalant use, availability of cigarettes in the home, depression, and 
    self-esteem.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('All CSV/tree_addhealth.csv')
dataset.info()

## check missing column or not
check_miss= dataset.isnull().any(axis=0)

## handling missing data for each column with its mode 
for col in range(25):
    if check_miss[col]== True:      ## finding mode(most frequent value)
        mode= dataset.iloc[:,col].value_counts().index[0]
        dataset.iloc[:,col]= dataset.iloc[:,col].fillna( mode )  
    else:
        pass

## extracting features and labels from dataset
df1= dataset.iloc[:, 0:7]
df2= dataset.iloc[:, 8:16]
features = pd.concat([df1, df2], axis=1) # adding dataframes columnwise
labels = dataset.iloc[:,7]

# Splitting the dataset into the Training set and Test set  
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
                train_test_split(features, labels, test_size = 0.25, random_state = 40)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
features_train = sc.fit_transform(features_train)
features_test = sc.transform(features_test)

# Fitting K-NN to the Training set
# When p = 1, for manhattan_distance (l1), and euclidean_distance (l2) for p = 2
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 7, p = 2) 
classifier.fit(features_train, labels_train)

#Calculate Class Probabilities
probability = classifier.predict_proba(features_test)

# Predicting the class labels
labels_pred = classifier.predict(features_test)
print(pd.DataFrame(zip(labels_test, labels_pred), columns= ['Actual','Predicted'] ))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels_test, labels_pred)
print(cm)

''' 
for n= 7 :
     [[18  0]
     [ 3  2]]
for n= 5 :
    [[16  2]
     [ 4  1]]
'''
# Accuracy Score / Confusion Matrix
from sklearn.metrics import accuracy_score  
print (accuracy_score(labels_test, labels_pred)*100) # 86.95 %

################################################################################
"""
  2.Build a classification tree model evaluation if an adolescent gets expelled or 
    not from school based on their Gender and violent behavior.
"""

dataset.info()
features = dataset.iloc[:, [0,16]]
labels = dataset.iloc[:,21]

# Splitting the dataset into the Training set and Test set  
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
                train_test_split(features, labels, test_size = 0.25, random_state = 40)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
features_train = sc.fit_transform(features_train)
features_test = sc.transform(features_test)

# Fitting K-NN to the Training set
# When p = 1, for manhattan_distance (l1), and euclidean_distance (l2) for p = 2
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 3, p = 2) 
classifier.fit(features_train, labels_train)

#Calculate Class Probabilities
probability = classifier.predict_proba(features_test)

# Predicting the class labels
labels_pred = classifier.predict(features_test)
print(pd.DataFrame(zip(labels_test, labels_pred), columns= ['Actual','Predicted'] ))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels_test, labels_pred)
print(cm)

''' n= 7
 [[22  0]
 [ 1  0]]
'''
# Accuracy Score / Confusion Matrix
from sklearn.metrics import accuracy_score  
print (accuracy_score(labels_test, labels_pred)*100)  # 95.65 %

##################################################################
"""
  3.Use random forest in relation to regular smokers as a target and explanatory 
    variable specifically with Hispanic, White, Black, Native American and Asian.
"""

dataset.info()      ## list of desired columns
col= list(range(1,7))+ [9] + list(range(14,21))+ list(range(22,25))  

features = dataset.iloc[:, col]
labels = dataset.iloc[:,7]

# Splitting the dataset into the Training set and Test set  
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
                train_test_split(features, labels, test_size = 0.25, random_state = 40)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
features_train = sc.fit_transform(features_train)
features_test = sc.transform(features_test)

# train the model
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=15, random_state=0)  
classifier.fit(features_train, labels_train)  

# Comparing the predicted and actual values
labels_pred = classifier.predict(features_test) 
my_frame= pd.DataFrame({'Actual':labels_test, 'Predicted':labels_pred})
print(my_frame)

# Evaluating score
from sklearn.metrics import confusion_matrix  
cm = confusion_matrix(labels_test, labels_pred)
print(cm)  

'''
fot 15 decision tree
[[18  0]
 [ 3  2]]
'''
# Accuracy Score / Confusion Matrix
from sklearn.metrics import accuracy_score  
print (accuracy_score(labels_test, labels_pred)*100)  # 86.95 %



