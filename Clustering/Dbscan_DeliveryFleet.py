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

# Compute DBSCAN
from sklearn.cluster import DBSCAN
db = DBSCAN(eps=1, min_samples=10).fit(features)

labels_pred = db.labels_ # belongs to which cluster id
print(labels_pred)

freq= np.unique(labels_pred, return_counts=True)
freq

# Using DBSCAN, 10 clusters are generated
col= ['red','blue','yellow', 'pink','black','brown','orange','gray','green','maroon']
marker= ['*','+','o','d','D','x','X','h','H',3]

## Visualising the clusters
for i in freq[0]:
    plt.scatter(features[labels_pred == i, 0], features[labels_pred == i, 1], \
                c = col[i], marker= marker[i]  )

plt.title('Clusters of Drivers using DBSCAN')
plt.xlabel('Distance_Feature')
plt.ylabel('Speed_Feature')
plt.legend()
plt.show()



