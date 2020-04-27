"""
Code Challenge:
dataset: BreadBasket_DMS.csv

Q1. In this code challenge, you are given a dataset which has data and time wise 
    transaction on a bakery retail store.
1. Draw the pie chart of top 15 selling items.
2. Find the associations of items where min support should be 0.0025, 
   min_confidence=0.2, min_lift=3.
3. Out of given results sets, show only names of the associated item from given result 
   row wise.
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from apyori import apriori

# Data Preprocessing
# Column names of the first row is missing, header - None
data = pd.read_csv('All CSV/BreadBasket_DMS.csv')

data.info()

"1. Draw the pie chart of top 15 selling items."
df= data['Item'].value_counts()
plt.pie(df[:15], labels= df.index[:15], autopct='%.0f%%')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

'''
2.Find the associations of items where min support should be 0.0025, 
   min_confidence=0.2, min_lift=3.
  
3.Out of given results sets, show only names of the associated item from given result 
   row wise.
'''

data.describe()

tr_min= data['Transaction'].min()
tr_max= data['Transaction'].max()

transactions = []
for i in range(tr_min, tr_max):
    lis= data[data['Transaction']== i]['Item']
    transactions.append(list(lis) )


# Training Apriori on the dataset
rules = apriori(transactions, min_support = 0.0025, min_confidence = 0.2, min_lift = 3)
print(type(rules))

# Visualising the results
results = list(rules)
print(len(results))

results[0]
results[0].items
results[0][0]

results[0].support 
results[0][1]  #--> support

results[0].confidence 
# at index = 2 we have ordered_statistics
results[0][2]
results[0][2][2]
results[0][2][0]
results[0][2][0][2]  #--> Confidence
results[0][2][0][3]  #--> Lift


ass_item=[]
for item in results:
    # first index of the inner list
    # Contains base item and add item
    pair = item[0]
    items = [x for x in pair]
    print("Rule: " + items[0] + " -> " + items[1])
    ass_item.append(items[1])
    
    #second index of the inner list
    print("Support: " + str(item[1]))

    #third index of the list located at 0th
    #of the third index of the inner list

    print("Confidence: " + str(item[2][0][2]))
    print("Lift: " + str(item[2][0][3]))
    print("=====================================")


print('Names of the Associated items:',set(ass_item) )




