
"""Here, we are building a decision tree to check if a person is hired or
 not based on certain predictors.

Import PastHires.csv File.

scikit-learn needs everything to be numerical for decision trees to work.

So, use any technique to map Y,N to 1,0 and levels of education to some scale of 0-2.

Build and perform Decision tree based on the predictors and see how accurate 
your prediction is for a being hired.

Now use a random forest of 10 decision trees to predict employment of specific 
candidate profiles:

Predict employment of a currently employed 10-year veteran, previous 
employers 4, went to top-tire school, having Bachelor's Degree without Internship.

Predict employment of an unemployed 10-year veteran, ,previous employers 4, 
didn't went to any top-tire school, having Master's Degree with Internship.

filename "PastHires.py"
"""
import pandas as pd

df = pd.read_csv("All CSV/PastHires.csv")
col = df.columns[[1,4,5,6]]

for i in col:
    df[i][df[i]=='Y'] = 1
    df[i][df[i]=='N']=0

d = {"BS":0,"MS":1,"PhD":2}
df["Level of Education"] = df["Level of Education"].map(d)

x = df[df.columns[:6]]
y = df["Hired"]

from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor(random_state=0)
dtr = dtr.fit(x,y)
D_pred = dtr.predict(x)

from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators=10)
rfr = rfr.fit(x,y)
R_pred = rfr.predict(x)

#Predict employment of an employed 10-year veteran
emp = rfr.predict([[10, 1, 4, 0, 1, 0]])

print ("\n")
print ("chances of getting hired for currently employed person is : "+str(emp[0]*100)+"%")
print ("\n")

#...and an unemployed 10-year veteran
un_emp = rfr.predict([[10, 0, 4, 1, 0, 1]])

print ("chances of getting hired for currently un-employed person is : "+str(un_emp[0]*100)+"%")

