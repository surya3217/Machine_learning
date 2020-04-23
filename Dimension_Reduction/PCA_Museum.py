"""
Q3 Data: "data.csv"

This data is provided by The Metropolitan Museum of Art Open Access
1. Visualize the various countries from where the artworks are coming.
2. Visualize the top 2 classification for the artworks
3. Visualize the artist interested in the artworks
4. Visualize the top 2 culture for the artworks
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
data = pd.read_csv('All CSV/data.csv')

# checking missing data
data.info()
data.isnull().any(axis= 0)

"1. Visualize the various countries from where the artworks are coming."
df1= data['Country'].value_counts()
index= list(range(20))
plt.bar(index, df1[:20] )
plt.xlabel('Countries', fontsize=15)
plt.ylabel('No. of count', fontsize=15)
plt.xticks(index, df1.index, fontsize=10, rotation=90)
plt.title('Artworks from Various Countries ')
plt.show()

"2. Visualize the top 2 classification for the artworks."
df2= data['Classification'].value_counts()
index= list(range(18))
plt.bar(index, df2 )
plt.xlabel('Artworks', fontsize=15)
plt.ylabel('No. of count', fontsize=15)
plt.xticks(index, df2.index, fontsize=10, rotation=90)
plt.title('Classification for the Artworks')
plt.show()

plt.pie(df2[:8], labels= df2.index[:8], autopct='%.0f%%')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

"3. Visualize the artist interested in the artworks."
df3= data['Artist Role'].value_counts()
index= list(range(20))
plt.bar(index, df3[:20] )
plt.xlabel('Artist Role', fontsize=15)
plt.ylabel('No. of count', fontsize=15)
plt.xticks(index, df3.index, fontsize=10, rotation=90)
plt.title("Artist's interest in the Artworks")
plt.show()

"4. Visualize the top 2 culture for the artworks."
df4= data['Culture'].value_counts()
index= list(range(20))
plt.bar(index, df4[:20] )
plt.xlabel('Culture', fontsize=15)
plt.ylabel('No. of count', fontsize=15)
plt.xticks(index, df4.index, fontsize=10, rotation=90)
plt.title("Culture for the Artworks")
plt.show()








