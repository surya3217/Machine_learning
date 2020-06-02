"""
Code Challenge:

Q1. (Create a program that fulfills the following specification.)

Program Specification:
Import breast_cancer.csv file.

This breast cancer database was obtained from the University of Wisconsin Hospitals,
Madison from Dr. William H. Wolberg.

Attribute Information: (class attribute has been moved to last column)

Sample Code Number(id number)----> represented by column A.
Clump Thickness (1 – 10) ----> represented by column B.
Uniformity of Cell Size(1 - 10)----> represented by column C.
Uniformity of Cell Shape (1 - 10)----> represented by column D.
Marginal Adhesion (1 - 10)----> represented by column E.
Single Epithelial Cell Size (1 - 10)----> represented by column F.
Bare Nuclei (1 - 10)----> represented by column G.
Bland Chromatin (1 - 10)----> represented by column H.
Normal Nuclei (1 - 10)----> represented by column I.
Mitoses (1 - 10)----> represented by column J.
Class: (2 for Benign and 4 for Malignant)----> represented by column K.
 
A Benign tumor is not a cancerous tumor and Malignant tumor is a cancerous tumor.

1.Impute the missing values with the most frequent values.
2.Perform Classification on the given data-set to predict if the tumor is 
  cancerous or not.
3.Check the accuracy of the model.
4.Predict whether a women has Benign tumor or Malignant tumor, 
  if her Clump thickness is around 6, uniformity of cell size is 2, Uniformity of
  Cell Shape is 5, Marginal Adhesion is 3, Bland Chromatin is 9, Mitoses is 4, 
  Bare Nuclei is 7 Normal Nuclei is 2 and Single Epithelial Cell Size is 2.

(you can neglect the id number column as it doesn't seem  a predictor column)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("All CSV/breast_cancer.csv")
dataset.info()
dataset.isnull().any(axis=0)

"Handle missing data with most frequent value"
freq_val= dataset['G'].value_counts().index
dataset['G']= dataset['G'].fillna(freq_val[0] )
dataset.info()

features = dataset.iloc[:, 1:-1].values
labels = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.3, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
features_train = sc.fit_transform(features_train)
features_test = sc.transform(features_test)

# Fitting Kernel SVM to the Training set
# kernels: linear, rbf and poly
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(features_train, labels_train)

# Predicting the Test set results
labels_pred = classifier.predict(features_test)

# Comparing the predicted and actual values
my_frame= pd.DataFrame(labels_pred, labels_test)
print(my_frame)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels_test, labels_pred)
print(cm)

# Model Score
score = classifier.score(features_test,labels_test)  ## 95.23 %
print(score*100)

"""
linear score: 95.23
rbf score   : 93.80
poly score  : 92.85
"""

'''
4.Predict whether a women has Benign tumor or Malignant tumor, 
  if her Clump thickness is around 6, uniformity of cell size is 2, Uniformity of
  Cell Shape is 5, Marginal Adhesion is 3, Bland Chromatin is 9, Mitoses is 4, 
  Bare Nuclei is 7 Normal Nuclei is 2 and Single Epithelial Cell Size is 2.
'''

x= np.array([6,2,5,3,2,7,9,2,4 ])
pred = classifier.predict([x])
print('Tumor is not cancerous.' if pred==2 else 'Tumor is cancerous.')
###########################################

import statsmodels.api as sm

features_obj = features[:, [0,1,2,3,4,5,6,7,8]]
features_obj = sm.add_constant(features_obj)
while (True):
    regressor_OLS = sm.OLS(endog = labels,exog =features_obj).fit()
    p_values = regressor_OLS.pvalues
    if p_values.max() > 0.05 :
        features_obj = np.delete(features_obj, p_values.argmax(),1)
    else:
        break

#print(features_obj) 
regressor_OLS.summary()

print("""From the above algorithm, we can say that 3 columns are not significant in computing: 
      1. Marginal Adhesion(column E)
      2. Single Epithelial Cell Size(column F)
      3. Mitoses(column J)
""")

#################################################################################

# Another Method


# Importing the libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import train_test_split as tts
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

# Importing the dataset
data = pd.read_csv('breast_cancer.csv')
x = data.iloc[:,1:-1].values
y = data.iloc[:,-1].values

# Handling Missing Values
imp = Imputer(missing_values='NaN', strategy='most_frequent')
x = imp.fit_transform(x)

# Splitting the Dataset
x_train, x_test, y_train, y_test = tts(x,y,random_state=0, test_size=0.25)

# Building SVM model
classifier =  SVC(kernel='linear', random_state=0)
classifier.fit(x_train, y_train)

# Predicting the values
Pred = classifier.predict(x_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, Pred)

# Model Score
score = classifier.score(x_test,y_test)

val = np.array([6,2,5,3,2,7,9,2,4]).reshape(1,-1)
val_Pred = classifier.predict(val)

print ("Accuracy of the Model : "+str(round(score*100,2))+"%")
print ("\n")

if (val_Pred==4):
    print ("Woman has Malignant Tumor")
else:
    print ("Woman has Benign Tumor")


