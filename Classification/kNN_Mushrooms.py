"""
Q1. (Create a program that fulfills the following specification.)
Import mushrooms.csv file

This dataset includes descriptions of hypothetical samples corresponding to 23 
species of gilled mushrooms in the Agaricus and Lepiota Family Mushroom drawn from 
The Audubon Society Field Guide to North American Mushrooms (1981). Each species is 
identified as definitely edible, definitely poisonous, or of unknown edibility and 
not recommended. This latter class was combined with the poisonous one.

Attribute Information:

1. classes: edible=e, poisonous=p (outcome)
2. cap-shape: bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s
3. cap-surface: fibrous=f,grooves=g,scaly=y,smooth=s
4. cap-color: brown=n, buff=b,cinnamon=c,gray=g,green=r,pink=p,purple=u,red=e,white=w,
   yellow=y
5. bruises: bruises=t, no=f
6. odor: almond=a,anise=l,creosote=c,fishy=y,foul=f,musty=m,none=n,pungent=p,spicy=s
7. gill-attachment: attached=a,descending=d,free=f,notched=n
8. gill-spacing: close=c,crowded=w,distant=d
9. gill-size: broad=b,narrow=n\
10. gill-color: black=k,brown=n,buff=b,chocolate=h,gray=g,green=r,orange=o,pink=p,
    purple=u,red=e,white=w,yellow=y
11. stalk-shape: enlarging=e,tapering=t
12. stalk-root: bulbous=b,club=c,cup=u,equal=e,rhizomorphs=z,rooted=r,missing=?
13. stalk-surface-above-ring: fibrous=f,scaly=y,silky=k,smooth=s
14. stalk-surface-below-ring: fibrous=f,scaly=y,silky=k,smooth=s
15. stalk-color-above-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,
    white=w,yellow=y
16. stalk-color-below-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,
    white=w,yellow=y
17. veil-type: partial=p,universal=u
18. veil-color: brown=n,orange=o,white=w,yellow=y
19. ring-number: none=n,one=o,two=t
20. ring-type: cobwebby=c,evanescent=e,flaring=f,large=l,none=n,pendant=p,sheathing=s,
    zone=z
21. spore-print-color: black=k,brown=n,buff=b,chocolate=h,green=r,orange=o,purple=u,
    white=w,yellow=y
22. population: abundant=a,clustered=c,numerous=n,scattered=s,several=v,solitary=y
23. habitat: grasses=g,leaves=l,meadows=m,paths=p,urban=u,waste=w,woods=d

1.Perform Classification on the given dataset to predict if the mushroom is edible or 
  poisonous w.r.t. it’s different attributes.
(you can perform on habitat, population and odor as the predictors)

class: label 0
odor: 5
population: 21
habitat: 22
    
2.Check accuracy of the model.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('All CSV/mushrooms.csv')
dataset.info()

features = dataset.iloc[:, [5,21,22]]
labels = dataset.iloc[:, 0]

# Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

for i in range(features.shape[1]):
    columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [-1])], \
                                       remainder='passthrough')
    features = np.array(columnTransformer.fit_transform(features), dtype = np.str)
    features = features[:, 1:]   ## removing redundant column

print(features)

## converting object dtype to float
features = features.astype('float64') 

d= {'p': 0, 'e': 1}
labels= labels.map(d).values

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
classifier = KNeighborsClassifier(n_neighbors = 5, p = 2) 
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
n= 5
[[ 965   10]
 [   0 1056]]
'''
# Accuracy Score / Confusion Matrix
from sklearn.metrics import accuracy_score  
print (accuracy_score(labels_test, labels_pred)*100) # 99.50 %




