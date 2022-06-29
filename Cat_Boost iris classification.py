#Iris dataset classification using CatBoost
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_wine
data=load_wine()
y=data.target
x=data.data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.7,random_state=0)
from catboost import CatBoostClassifier
clf=CatBoostClassifier(learning_rate=0.3,iterations=100)
clf.fit(x_train,y_train)
from sklearn.metrics import accuracy_score,confusion_matrix
y_pred=clf.predict(x_test)
print("accuracy score :",accuracy_score(y_test,y_pred))
cm=confusion_matrix(y_test,y_pred)
import seaborn as sns
sns.heatmap(cm,annot=True)
plt.show()