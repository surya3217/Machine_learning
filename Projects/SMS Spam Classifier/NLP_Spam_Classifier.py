'''
The SMS Spam Corpus v.0.1 is a set of SMS tagged messages that have been collected
for SMS Spam research. It contains two collections of SMS messages in English of 
1084 and 1319 messages, tagged according being legitimate (ham) or spam.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer   # For Stemming

## opening the file and making the DataFrame to analyze
data = pd.read_csv('spam.csv', encoding = "latin-1") 
data = data[['v1', 'v2']]
data = data.rename(columns = {'v1': 'label', 'v2': 'text'})

# check missing column or not
data.isnull().any(axis=0)

#pie chart
data_spam= data['label'].value_counts()
plt.pie(data_spam, labels= data_spam.index, autopct='%.0f%%')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

#perform row wise noise removal and stemming
#let's do it on just first row data
import re
print(data['text'][0])

"""
Search through regex for special character set , using the substitute function 
substitute the regex with space ' ' 
[^a-zA-Z ] finds those which does not belong to a to z or A to Z
"""
#lem = WordNetLemmatizer()  # Another way of finding root word
ps = PorterStemmer()

# Now do the same for every row in dataset. run to loop for all rows
# Add into this bigger list
def clean_messages(text):
 
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [word for word in text if not word in set(stopwords.words('english'))]
    
    text = [ps.stem(word) for word in text]    
    text = ' '.join(text) 
    return text

data['text'] = data['text'].apply(clean_messages)

## Applying the algorithms
#from sklearn.feature_extraction.text import CountVectorizer
#cv = CountVectorizer(max_features = 2000)    # 2000 columns

# it is known as sparse matrix of the features ND Array
#features = cv.fit_transform(data['text']).toarray()  
#labels = data.iloc[:, 0].values

features = data['text'].values  
labels = data.iloc[:, 0].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
            train_test_split(features, labels, test_size = 0.20, random_state = 0)

'''
we will use the TF-IDF vectorizer (Term Frequency — Inverse Document Frequency), 
a similar embedding technique which takes into account the importance of each term
to document.

TF-IDF vectorizer was chosen for its simplicity and efficiency in vectorizing documents such as text messages.

TF-IDF vectorizes documents by calculating a TF-IDF statistic between the document 
and each term in the vocabulary.

A tokenizer splits documents into tokens (thus assigning each token to its own term)
based on white space and special characters.

For example, the phrase what’s going on might be split into what, ‘s, going, on.
'''
# training the vectorizer 
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
features_train = vectorizer.fit_transform(features_train)
#############################################################
'''
The C term is used as a regularization to influence the objective function.
A larger value of C typically results in a hyperplane with a smaller margin as it
gives more emphasis to the accuracy rather than the margin width. 
'''
from sklearn import svm
svm = svm.SVC(C=1000)
svm.fit(features_train, labels_train)

from sklearn.metrics import confusion_matrix
features_test = vectorizer.transform(features_test)
labels_pred = svm.predict(features_test)

cm= confusion_matrix(labels_test, labels_pred)
print(cm)
'''
[[947   2]
 [ 34 132]]
'''
######################################################

# DecissionTreeClassifier
from sklearn.tree import DecisionTreeClassifier  
classifier = DecisionTreeClassifier()  
classifier.fit(features_train, labels_train)

labels_pred = classifier.predict(features_test) 

# Comparing the predicted and actual values
my_frame= pd.DataFrame({'Actual':labels_test, 'Predicted':labels_pred})
print(my_frame)

# Evaluating score
cm2 = confusion_matrix(labels_test, labels_pred)
print(cm2)  
'''
[[930  19]
 [ 31 135]]
'''
# Model Score = 98.02 % times out of 100 model prediction was RIGHT
print( (cm2[0][0] + cm2[1][1]) / np.sum(cm2) )
########################################################

#test message

#im donee. come pick me up
#winner$$$$ SMS REPLY "WIN"
#whats the matter wit u
#Come to think of it,i never got a spam text message before

def pred(msg):    
    msg = vectorizer.transform([msg])    
    prediction = svm.predict(msg)    
    return prediction[0]

clean_messages('im donee. come pick me up')
pred('im donee. come pick me up')



