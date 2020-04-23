"""
Q2.The iris data set consists of 50 samples from each of three species of Iris 
flower (Iris setosa, Iris virginica and Iris versicolor).

Four features were measured from each sample: the length and the width 
of the sepals and petals, in centimetres (iris.data).
Import the iris dataset already in sklearn module using the following command

from sklearn.datasets import load_iris
iris = load_iris()
iris=iris.data

Reduce dimension from 4-d to 2-d and perform clustering to distinguish the 3 species.
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

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
features = sc.fit_transform(features)

# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
features = pca.fit_transform(features)

# How much is the loss and how much we are able to retain the information
explained_variance = pca.explained_variance_ratio_
print(explained_variance)
# first paramater (PC1) is holding 72.9 % of the 4D data
# second parameter (PC2) is holding 22.8 % of the 4D data

# Scatter all these data points on the matplotlib
plt.scatter(features[:,0], features[:,1])
plt.show()


"Fitting K-Means to the dataset using 3 clusters for three species"
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 0)

pred_cluster = kmeans.fit_predict(features) # We have only passed features 
print(pred_cluster)    # Its the cluster id with 0, 1, and 2 

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels, pred_cluster)
print(cm)

# Model Score = 70.4 % times out of 100 model prediction was RIGHT
print( (cm[0][0]+ cm[1][1]) / np.sum(cm) )

# There are three points for 3 centroid
print(kmeans.cluster_centers_)  
print(kmeans.cluster_centers_[:, 0]) # x 
print(kmeans.cluster_centers_[:, 1]) # y 

clusters= ['Setosa', 'Versicolor', 'Virginica']

# Visualising the clusters 
plt.scatter(features[pred_cluster == 0, 0], features[pred_cluster == 0, 1], c = 'blue', label = clusters[0])
plt.scatter(features[pred_cluster == 1, 0], features[pred_cluster == 1, 1], c = 'red', label = clusters[1])
plt.scatter(features[pred_cluster == 2, 0], features[pred_cluster == 2, 1], c = 'green', label = clusters[2])
#plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c = 'black', label = 'Centroids')

plt.title('Clusters of species of Iris flower')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()
############################################################

"Fitting DBSCAN to the dataset"
from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=0.5, metric='euclidean', min_samples=5)
print (dbscan)

dbscan.fit(iris.data)  ## applying algo for 4D data
dbscan.labels_

# Visualising the clusters 
# as data is in 4D space, we need to apply PCA for 2D ploting
from sklearn.decomposition import PCA
pca = PCA(n_components=2).fit(iris.data)
pca_2d = pca.transform(iris.data)

#alternative way, fit_transform
#pa = PCA(n_components = 2)
#pca_2d =  pca.fit_transform(iris.data)

for i in range(0, pca_2d.shape[0]):
    if dbscan.labels_[i] == 0:
        c1 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='r', marker='+')
    elif dbscan.labels_[i] == 1:
        c2 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='g', marker='o')
    elif dbscan.labels_[i] == -1:
        c3 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='b', marker='*')

plt.legend([c1, c2, c3], ['Cluster 1', 'Cluster 2','Noise'])
plt.title('DBSCAN finds 2 clusters and noise')
plt.savefig("dbscan.jpg")
plt.show()

pca.components_





