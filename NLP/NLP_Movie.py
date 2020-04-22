"""
Q3 Program Specification

Import movie.csv file

There are two categories: Pos (reviews that express a positive or favorable sentiment) 
and Neg (reviews that express a negative or unfavorable sentiment). 
For this assignment, we will assume that all reviews are either positive or negative; 
there are no neutral reviews.

Perform sentiment analysis on the text reviews to determine whether its positive 
or negative and build confusion matrix to determine the accuracy.
"""

import pandas as pd
import numpy as np
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer   # For Stemming

# opening the file and making the DataFrame to analyze
dataset = pd.read_csv('All CSV/movie.csv') 

# check missing column or not
dataset.isnull().any(axis=0)

"""
Apply the process to one line of text
This will help in understanding the below logic
"""
#perform row wise noise removal and stemming
#let's do it on just first row data

import re
print(dataset['text'][0])

"""
Search through regex for special character set , using the substitute function 
substitute the regex with space ' ' 
[^a-zA-Z ] finds those which does not belong to a to z or A to Z
"""
review = re.sub('[^a-zA-Z]', ' ', dataset['text'][0])
print(review)

review = review.lower()
print(review)

review = review.split()
print(review)

#We need to check whether it is a stopword, if YES then remove it
review = [word for word in review if not word in set(stopwords.words('english'))]
print(review)


# lem = WordNetLemmatizer()  # Another way of finding root word
ps = PorterStemmer()

review = [ps.stem(word) for word in review]
print(review)

review = ' '.join(review)
print(review)

# Now do the same for every row in dataset. run to loop for all rows
# Add into this bigger list
corpus = []
 
for i in range(0, 2000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['text'][i])
    review = review.lower()
    review = review.split()
    review = [word for word in review if not word in set(stopwords.words('english'))]
    
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review]
    
    review = ' '.join(review)    
    corpus.append(review)

print(corpus)
print(len(corpus))


# Applying the algorithms
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)    # 1500 columns

# it is known as sparse matrix of the features ND Array
features = cv.fit_transform(corpus).toarray()  
labels = dataset.iloc[:, 0]

d= {'Pos': 1, 'Neg': 0}
labels= labels.map(d).values

print(features.shape)
print(labels.shape)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
            train_test_split(features, labels, test_size = 0.25, random_state = 0)


# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(features_train, labels_train)

# Predicting the Test set results
labels_pred = classifier.predict(features_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_nb = confusion_matrix(labels_test, labels_pred)
print(cm_nb)

# Model Score = 70.4 % times out of 100 model prediction was RIGHT
print( (cm_nb[0][0]+ cm_nb[1][1]) / np.sum(cm_nb) )
####################################################################

# train the model
from sklearn.ensemble import RandomForestClassifier
Classifier = RandomForestClassifier(n_estimators= 300, random_state=0)  
Classifier.fit(features_train, labels_train)  

# Predicting the Test set results
labels_pred = Classifier.predict(features_test)

from sklearn.metrics import confusion_matrix
cm_rf = confusion_matrix(labels_test, labels_pred)
print(cm_rf)

# Model Score = 82.2 % times out of 100 model prediction was RIGHT
print( (cm_rf[0][0]+ cm_rf[1][1]) / np.sum(cm_rf) )

print('By the above analysis, we can say that "RandomForestClassifier" performs better than "Naive Bayes".')



