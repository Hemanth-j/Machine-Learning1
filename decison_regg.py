#Decision tree with regression clearly explained
import matplotlib.pyplot as plt
import pandas as pd
df=pd.read_csv("C:\\Users\\HEMANTH\\Downloads\\archive\decision-tree-regression-dataset.csv")
x=df.iloc[:,[0]].values
y=df.iloc[:,[1]].values
print(x,y)
from sklearn.tree import DecisionTreeRegressor
clf=DecisionTreeRegressor()
clf.fit(x,y)
print(clf.score(x,y))
from sklearn import tree
plt.figure(figsize=(14,9))
_=tree.plot_tree(clf,filled=True,rounded=True)
print(clf.predict([[10]]))
print(clf.predict([[5]]))
print(clf.predict([[20]]))
plt.show()
