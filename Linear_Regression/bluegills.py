"""
Q. (Create a program that fulfills the following specification.)
bluegills.csv

1. How is the length of a bluegill fish related to its age?

In 1981, n = 78 bluegills were randomly sampled from Lake Mary in Minnesota. The
researchers (Cook and Weisberg, 1999) measured and recorded the following data (Import 
bluegills.csv File)

Response variable(Dependent): length (in mm) of the fish
Potential Predictor (Independent Variable): age (in years) of the fish

2. How is the length of a bluegill fish best related to its age? (Linear/Quadratic nature?)
3. What is the length of a randomly selected five-year-old bluegill fish? Perform 
   polynomial regression on the dataset.

NOTE: Observe that 80.1% of the variation in the length of bluegill fish is reduced by 
taking into account a quadratic function of the age of the fish.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing the dataset
data = pd.read_csv('All CSV/bluegills.csv')
data.info()

# Check for missing data column wise
data.isnull().any(axis=0)

features= data.iloc[:, :1].values
labels= data.iloc[:, 1 ].values

# Dataset is small so no splitting
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test= train_test_split(features, labels, test_size= 0.2, random_state= 0)

# Fitting Multiple Linear Regression to the Training set
# Whether we have Univariate or Multivariate, class is LinearRegression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(features_train, labels_train)

# y = ax1 + bx2 + c
# Here a, b and c are the coefficients and d is the intercept
print(regressor.intercept_)   
print (regressor.coef_)
print ('Training Score:',regressor.score(features_train, labels_train)*100) # 74.44 %

# Predicting the Test set results
pred = regressor.predict(features_test)
print (pd.DataFrame(zip(labels_test, pred )))
print ('Testing Score:',regressor.score(features_test, pred )*100)

# Visualising the Linear Regression results
plt.scatter(features, labels, color = 'red')
plt.plot(features, regressor.predict(features), color = 'blue')
plt.title('Linear Regression')
plt.xlabel('Age')
plt.ylabel('Length')
plt.show()

print(regressor.predict([[5]]))  # 175.50


###############################################################################


# After seeing the visual, its seems that the predictions will be poor
# Once the age increases
# Actual line should be a polynomial line
# What should be the degree of the polynomial function 
# Its a hit and trail method and visualize it to see the curve


# We need to convert the feature(x) into 5 degree format
# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_object = PolynomialFeatures(degree = 5)
print(features.shape)

features_poly = poly_object.fit_transform(features)
print(features_poly)
print(features_poly.shape) # x0 x1 x2 x3 x4 x5    

# Algo is same for Polynomial Regression, its only the data format is changed
poly_reg = LinearRegression()
poly_reg.fit(features_poly, labels)

print ("Predicting result with Polynomial Regression")

# Visualising the Polynomial Regression results
plt.scatter(features, labels, color = 'red')
plt.plot(features, poly_reg.predict(features_poly), color = 'blue')
plt.title('Polynomial Regression')
plt.xlabel('Year')
plt.ylabel('Claims Paid')
plt.show()

print ('Training Score:', poly_reg.score(features_poly, labels)*100) # 80.35 %

# Predicting the Test set results
pred = poly_reg.predict(features_poly)
print (pd.DataFrame(zip(labels, pred )))


print(lin_reg_2.predict(poly_object.fit_transform([[5]]) ) ) # 166.125
print("So polynomial Linear Regression give better result at degree of 5 as its score is better than simple linear regression.")


