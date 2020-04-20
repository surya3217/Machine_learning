"""
Q2. This famous classification dataset first time used in Fisher’s classic 1936 paper, 
The Use of Multiple Measurements in Taxonomic Problems. 
Iris dataset is having 4 features of iris flower and one target class.

The 4 features are:
SepalLength(cm), SepalWidth(cm), PetalLength(cm), PetalWidth(cm)

The target class: The flower species type is the target class and it having 3 types
Setosa, Versicolor, Virginica

The idea of implementing svm classifier in Python is to use the iris features to train
an svm classifier and use the trained svm model to predict the Iris species type. 
To begin with let’s try to load the Iris dataset.
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# loading the data from dataset
from sklearn.datasets import load_iris
iris = load_iris()
features= iris.data
labels= iris.target

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.3, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
features_train = sc.fit_transform(features_train)
features_test = sc.transform(features_test)

# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
features_train = pca.fit_transform(features_train)
features_test = pca.transform(features_test)

"SVM ( SVC for classification and SVR for Regression )"
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
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

'''
[[16  0  0]
 [ 0 15  3]
 [ 0  1 10]]
'''
# Model accuracy 
print( (cm[0][0] + cm[1][1]) / np.sum(cm))  ## 68.88 %

# Model Score
score = classifier.score(features_test,labels_test) ## 91.11 %
print(score)
##################################################################

#Visualization Way New
x_min, x_max = features_train[:, 0].min() - 1, features_train[:, 0].max() + 1
y_min, y_max = features_train[:, 1].min() - 1, features_train[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

# Obtain labels for each point in mesh using the model.
# ravel() is equivalent to flatten method.
# np.c_ : Translates slice objects to concatenation along the second axis.
# data dimension must match training data dimension, hence using ravel
Z = classifier.predict( np.c_[xx.ravel(), yy.ravel()] ).reshape(xx.shape)

# Plot the points
plt.plot(features_test[labels_test == 0, 0], features_test[labels_test == 0, 1], 'ro', label='Class 0')
plt.plot(features_test[labels_test == 1, 0], features_test[labels_test == 1, 1], 'bo', label='Class 1')
plt.plot(features_test[labels_test == 2, 0], features_test[labels_test == 2, 1], 'go', label='Class 2')

#plot the decision boundary
plt.contourf(xx, yy, Z, alpha=1.0)
plt.show()




