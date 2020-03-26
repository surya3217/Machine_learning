#kNN

# Open knn_class_prediction.jpg
"""
We have a training data that has 2 categories
that means label has either A or B
We are not concerned about the features at present
If we scatter the points on a graph
For testing we have a GREEN star
We need to predict the class or category of the GREEN STAR
Whether it will be A or B
This is the problem statement
This is a classification problem

There are a lot of friends of Sandeep who has two opinion.
Should we go for a movie today ( A means YES and B means NO )
Will Sandeep go for Movie ?

How this problem will be solved by kNN !!
What is the distance of the predicted point with the other points ?
What are the neareast 3 data points
2 belongs to Class B and 1 from Class A
Sandeep will go with the majority !!


Now check for the nearest 5 data points. ( Try to take ODD so that there is not tie)
This is the kNN Algorithm


In this algorithm Distance is the major factor.
In trigonometry how do we calculate distance between 2 points ?
p1 = (x1, y1)
p2 = (x2, y2)

Euclidean Distance ( Arial Distance )
d = square root of ( (x2 - x1)**2 + (y2 - y1)**2 )

There is one more distance ( based on Absolute values )
Manhatttan Distance ( NewYork Central Area is Manhattan, all road crosses at 90 )
D = |x2 - x1| + |y2 - y1|

Both of these distances are calculated from originally formula known as Minkoski Distance

Refer to https://en.wikipedia.org/wiki/Minkowski_distance

Explaiin the formula with 
X is the set of n dimension x points
Y is the set of n dimension y points

Lets assume 

p = 1 , then its Manhattan Distance

p = 2 , then its Euclidean Distance

So while using the kNN algo, we need to tell how to find the distance
"""

# Open caesarian.csv 

import sklearn as sk  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Last column marks whether it was caesarian or not ( 1 or 0 )
df = pd.read_csv('All CSV/caesarian.csv')  
print (df.head())


#get features and labels for our dataset

labels = df.iloc[:,5].values 
features = df.iloc[:,:-1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.25, random_state = 41)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
features_train = sc.fit_transform(features_train)
features_test = sc.transform(features_test)


# Fitting Logistic Regression to the Training set
from sklearn.neighbors import KNeighborsClassifier

# When p = 1, this is equivalent to using manhattan_distance (l1), 
# and euclidean_distance (l2) for p = 2
classifier = KNeighborsClassifier(n_neighbors = 5, p = 2) 


classifier.fit(features_train, labels_train)

#Calculate Class Probabilities
probability = classifier.predict_proba(features_test)

# Predicting the class labels
labels_pred = classifier.predict(features_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels_test, labels_pred)

print(cm)
# You cannot create confusion Matrix in case of Regression
# Since u cannot compare 2 continuous value


"""
Classification problem is always like drawing a decission boundary

LR is a linear Classifer, 
that means the decission boundary will always be straight line

kNN is a non linear Classifier,
that means the decission boundary will always be nonlinear in nature
"""

# Logistic Regression Visualisation ( Classification)
# Explain the viualisation code later
# If Sandeep age is 20 and he is earning 50000 salary, wuill he clikc the ad


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
# Last column, whether you have clicked the Ad or no
dataset = pd.read_csv('All CSV/Social_Network_Ads.csv')
features = dataset.iloc[:, :-1].values
labels = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.25, random_state = 40)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
features_train = sc.fit_transform(features_train)
features_test = sc.transform(features_test)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(features_train, labels_train)

#Calculate Class Probabilities
probability = classifier.predict_proba(features_test)

# Predicting the class labels
labels_pred = classifier.predict(features_test)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels_test, labels_pred)
print(cm)


  
"""
The purpose of meshgrid function in Python is to create a rectangular grid out 
of an array of x values and an array of y values. 
Suppose you want to create a grid where you have a point at each integer value 
between 0 and 4 in both the x and y directions.
"""
# X axis = AGE
# Y axis = Salary 

x_min, x_max = features_train[:, 0].min() - 1, features_train[:, 0].max() + 1
y_min, y_max = features_train[:, 1].min() - 1, features_train[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
# Obtain labels for each point in mesh using the model.
Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the points
plt.plot(features_test[labels_test == 1, 0], features_test[labels_test == 1, 1], 'bo', label='Class 2')
plt.plot(features_test[labels_test == 0, 0], features_test[labels_test == 0, 1], 'ro', label='Class 1')

plt.contourf(xx, yy, Z, alpha=0.3)
plt.show()

# Try to identify the 3 points in the graph according to the CM


# K-Nearest Neighbors (kNN) Visualisation 
# Explain the viualisation code later
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('All CSV/Social_Network_Ads.csv')
features = dataset.iloc[:, :-1].values
labels = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.25, random_state = 40)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
features_train = sc.fit_transform(features_train)
features_test = sc.transform(features_test)

# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, p = 2) #When p = 1, this is equivalent to using manhattan_distance (l1), and euclidean_distance (l2) for p = 2
classifier.fit(features_train, labels_train)

#Calculate Class Probabilities
probability = classifier.predict_proba(features_test)

# Predicting the class labels
labels_pred = classifier.predict(features_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels_test, labels_pred)
print(cm)


#Visualization
# Plot the decision boundary. For that, we will assign a color to each   

x_min, x_max = features_train[:, 0].min() - 1, features_train[:, 0].max() + 1
y_min, y_max = features_train[:, 1].min() - 1, features_train[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
# Obtain labels for each point in mesh using the model.
# ravel() is equivalent to flatten method.
# data dimension must match training data dimension, hence using ravel
Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# Plot the points
plt.plot(features_test[labels_test == 0, 0], features_test[labels_test == 0, 1], 'ro', label='Class 1: No')
plt.plot(features_test[labels_test == 1, 0], features_test[labels_test == 1, 1], 'bo', label='Class 2: Yes')
#plot the decision boundary
plt.contourf(xx, yy, Z, alpha=.5)

plt.show()


# kNN as compared to LR has better result 
# kNN is a non linear classifier


# It is immposible to tell seeing the data, whether LR or kNN will give better results
# You have to apply both the algo and comapre their cm
# We might have to apply SVM or Naive Bayes and then compare all

"""
https://stackoverflow.com/questions/36013063/what-is-purpose-of-meshgrid-in-python
https://www.kaggle.com/arthurtok/decision-boundaries-visualised-via-python-plotly
https://stackoverflow.com/questions/20045994/how-do-i-plot-the-decision-boundary-of-a-regression-using-matplotlib
https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
"""


"""
Code Challenge

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

Perform Classification on the given dataset to predict if the mushroom is edible or 
poisonous w.r.t. it’s different attributes.

(you can perform on habitat, population and odor as the predictors)

class: label 0
odor: 5
population: 21
habitat: 22
    
     Check accuracy of the model.
#####################################################


Q2. (Create a program that fulfills the following specification.)

For this Code Challenge, The National Longitudinal Study of Adolescent to Adult 
Health (Add Health) data set, an ongoing (longitudinal) survey study that began in 
the mid-1990s is used. The project website URL is:

http://www.cpc.unc.edu/projects/addhealth/.

This large data set is available online from the University of North Carolina’s 
Carolina Population Center, http://www.cpc.unc.edu/projects/addhealth/data.

Import tree_addhealth.csv

The attributes are:

1.BIO_SEX: 1 = male 0 = female    
2.HISPANIC: 1=Yes,0=No    
3.WHITE : 1=Yes,0=No
4.BLACK : 1=Yes,0=No          
5.NAMERICAN: 1=Yes,0=No                      
6.ASIAN: 1=Yes,0=No                      
7.ALCEVR1: ever drank alcohol(1=Yes,0=No)
8.marever1: ever smoked marijuana(1=Yes,0=No)    
9.cocever1: ever used cocaine(1=Yes,0=No)                
10.inhever1: ever used inhalants(1=Yes,0=No)             
11.cigavail: cigarettes available in home(1=Yes,0=No)
12.PASSIST: parents or public assistance(1=Yes,0=No)
13.EXPEL1: ever expelled from school(1=Yes,0=No)
14.TREG1: Ever smoked regularly(1=Yes,0=No)

#Explanatory Variables:

1.Age
2.ALCPROBS1:alcohol problems 0-6
3.DEP1: depression scale
4.ESTEEM1: self esteem scale       
5.VIOL1:violent behaviour scale
6.DEVIANT1: deviant behaviour scale     
7.SCHCONN1: school connectedness scale       
8.GPA1: gpa scale  4 points)
9.FAMCONCT: family connectedness scale       
10.PARACTV:parent activities scale
11.PARPRES:parental presence scale

    Build a classification tree model evaluating if an adolescent would smoke 
    regularly or not based on: gender, age, (race/ethnicity) Hispanic, White, Black, 
    Native American and Asian, alcohol use, alcohol problems, marijuana use, cocaine 
    use, inhalant use, availability of cigarettes in the home, depression, and 
    self-esteem.

    Build a classification tree model evaluation if an adolescent gets expelled or 
    not from school based on their Gender and violent behavior.
    Use random forest in relation to regular smokers as a target and explanatory 
    variable specifically with Hispanic, White, Black, Native American and Asian.

(Please make confusion matrix and also check accuracy score for each and every 
section)

"""


