"""
Q2. (Create a program that fulfills the following specification.)
tshirts.csv

T-Shirt Factory:
You own a clothing factory. You know how to make a T-shirt given the height and 
weight of a customer.

You want to standardize the production on three sizes: small, medium, and large. 
How would you figure out the actual size of these 3 types of shirt to better 
fit your customers?

Import the tshirts.csv file and perform Clustering on it to make sense out of 
the data as stated above.
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset (Bivariate Data Set with 3 Clusters)
dataset = pd.read_csv('All CSV/tshirts.csv')
dataset.info()   ## no missing data

features = dataset.iloc[:, [1, 2]].values

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

# Visualising the clusters
plt.scatter(features[pred_cluster == 0, 0], features[pred_cluster == 0, 1], c = 'blue', label = 'Medium')
plt.scatter(features[pred_cluster == 1, 0], features[pred_cluster == 1, 1], c = 'red', label = 'Large')
plt.scatter(features[pred_cluster == 2, 0], features[pred_cluster == 2, 1], c = 'green', label = 'Small')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c = 'black', label = 'Centroids')

plt.title('Clusters of Customers')
plt.xlabel('Height (inches)')
plt.ylabel('Weight (pounds)')
plt.legend()
plt.show()

# features falls with cluster id 0 and 1
print('Customer with \nsmall size T-shirt:',len(features[pred_cluster == 2]))
print('Medium size T-shirt:',len(features[pred_cluster == 0]))
print('Big size T-shirt:',len(features[pred_cluster == 1]))

print('So after performing the algorithm, we have found some conclusions:')
print('T-shirt size should be decided acoording to the table paramenters:\n')
df= pd.DataFrame(kmeans.cluster_centers_ , columns= \
                 ['Height (inches)', 'Weight (pounds)'], index= ['Medium','Large','Small'] )
print(df.sort_values('Height (inches)'))




