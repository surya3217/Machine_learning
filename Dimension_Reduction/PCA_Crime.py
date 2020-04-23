"""
Q1. Import Crime.csv File.
    Perform dimension reduction and group the cities using k-means based on 
    Rape, Murder and assault predictors.
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('All CSV/crime_data.csv')

# label is the customer categorories ( 1,2,3) who will like the wine
features = dataset.iloc[:, [1,2,4]].values

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
# first paramater (PC1) is holding 78.6 % of the 13D data
# second parameter (PC2) is holding 15.2 % of the 13D data

# Scatter all these data points on the matplotlib
plt.scatter(features[:,0], features[:,1])
plt.show()


# Fitting K-Means to the dataset using 3 clusters for three desired sizes
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 0)

pred_cluster = kmeans.fit_predict(features) # We have only passed features 
print(pred_cluster)    # Its the cluster id with 0, 1, and 2 


# There are three points for 3 centroid
print(kmeans.cluster_centers_)  
print(kmeans.cluster_centers_[:, 0]) # x 
print(kmeans.cluster_centers_[:, 1]) # y 

# Visualising the clusters Rape, Murder and assault
plt.scatter(features[pred_cluster == 0, 0], features[pred_cluster == 0, 1], c = 'blue', label = 'Murder')
plt.scatter(features[pred_cluster == 1, 0], features[pred_cluster == 1, 1], c = 'red', label = 'Assault')
plt.scatter(features[pred_cluster == 2, 0], features[pred_cluster == 2, 1], c = 'green', label = 'Rape')
#plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c = 'black', label = 'Centroids')

plt.title('Clusters of Crimes')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()


df= dataset.agg(np.sum)
index= list(range(4))
plt.bar(index, df[1:] )
plt.ylabel('No. of count', fontsize=15)
plt.xticks(index, df.index[1:], fontsize=10)
plt.title('Crimes in United States')
plt.show()

