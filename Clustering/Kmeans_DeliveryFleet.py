"""
Q1. (Create a program that fulfills the following specification.)
Import deliveryfleet.csv file

Here we need, Two driver features: mean distance driven per day (Distance_feature) 
and the mean percentage of time a driver was >5 mph over the speed limit (speeding_feature).

1.Perform K-means clustering to distinguish urban drivers and rural drivers.
2.Perform K-means clustering again to further distinguish speeding drivers 
  from those who follow speed limits, in addition to the rural vs. urban division.
  Label accordingly for the 4 groups.
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset (Bivariate Data Set with 3 Clusters)
dataset = pd.read_csv('All CSV/deliveryfleet.csv')
dataset.info()   ## no missing data

features = dataset.iloc[:, [1, 2]].values

# Scatter all these data points on the matplotlib
plt.scatter(features[:,0], features[:,1])
plt.show()

# Fitting K-Means to the dataset using 3 clusters
from sklearn.cluster import KMeans
# Since we have seen the visual, we have told the algo to make 3 cluster
kmeans = KMeans(n_clusters = 2, init = 'k-means++', random_state = 0)

pred_cluster = kmeans.fit_predict(features) # We have only passed features 
print(pred_cluster)    # Its the cluster id with 0, 1, and 2 

# There are four points for 4 centroid
print(kmeans.cluster_centers_)  
print(kmeans.cluster_centers_[:, 0]) # x 
print(kmeans.cluster_centers_[:, 1]) # y 

# Visualising the clusters
plt.scatter(features[pred_cluster == 0, 0], features[pred_cluster == 0, 1], c = 'blue', label = 'Rural Drivers')
plt.scatter(features[pred_cluster == 1, 0], features[pred_cluster == 1, 1], c = 'red', label = 'Urban Drivers')

plt.title('Clusters of Drivers')
plt.xlabel('Distance_Feature')
plt.ylabel('Speed_Feature')
plt.legend()
plt.show()

# features falls with cluster id 0 and 1
print('Number of Rural Drivers:',len(features[pred_cluster == 0]))
print('Number of Urban Drivers:',len(features[pred_cluster == 1]))

area = ['Rural', 'Urban']
vehicles = [len(features[pred_cluster == 0]), len(features[pred_cluster == 1])]
plt.bar(area,vehicles)
plt.title('Number of Vehicles')
plt.show()

#####################################################################
"""
2.Perform K-means clustering again to further distinguish speeding drivers 
  from those who follow speed limits, in addition to the rural vs. urban division.
"""

#mean= np.mean(features[:,1])
#f= lambda x: (mean*100)/x
#mean_per= f(features[:,1])

# Since we have seen the visual, we have told the algo to make 4 cluster
kmeans2 = KMeans(n_clusters = 4, init = 'k-means++', random_state = 0)

pred_cluster = kmeans2.fit_predict(features) # We have only passed features 
print(pred_cluster)    # Its the cluster id with 0, 1, and 2 

# There are four points for 4 centroid
print(kmeans2.cluster_centers_)  
print(kmeans2.cluster_centers_[:, 0]) # x 
print(kmeans2.cluster_centers_[:, 1]) # y 

# Visualising the clusters
plt.scatter(features[pred_cluster == 0, 0], features[pred_cluster == 0, 1], c = 'blue', label = 'Rural, Follow Speed')
plt.scatter(features[pred_cluster == 1, 0], features[pred_cluster == 1, 1], c = 'red', label = 'Urban, Follow Speed')
plt.scatter(features[pred_cluster == 2, 0], features[pred_cluster == 2, 1], c = 'green', label = 'Urban, Over Speed')
plt.scatter(features[pred_cluster == 3, 0], features[pred_cluster == 3, 1], c = 'brown', label = 'Rural, Over Speed')
#plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c = 'yellow', label = 'Centroids')

plt.title('Clusters of datapoints')
plt.xlabel('Distance_Feature')
plt.ylabel('Speed_Feature')
plt.legend()
plt.show()


data = [[len( features[pred_cluster == 0, 0]), len( features[pred_cluster == 1, 0]) ],
        [len( features[pred_cluster == 3, 0]), len( features[pred_cluster == 2, 0]) ] ]

# data= [[2773, 696], 
#        [427, 104]]

X = np.arange(2)
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(X + 0.00, data[0], color = 'b', width = 0.25)
ax.bar(X + 0.25, data[1], color = 'g', width = 0.25)
plt.legend()
#plt.xlabel('Distance_Feature')
plt.xticks(['a','b','c','d'])
plt.ylabel('Speed_Feature')





