## Linear SVM using iris dataset

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn import svm
import seaborn as sns
from sklearn.preprocessing import StandardScaler
std=StandardScaler()
data=load_iris()
df=pd.DataFrame(data.data,columns=data.feature_names)
df["Target"]=data.target
#print(df.head(100))
x=df.drop("Target",axis=1)
y=df.Target
x_train,x_test,Y_train,y_test=train_test_split(x,y,train_size=.8)
#std_x=std.fit_transform(x_train)
#print(x_train.shape,x_test.shape)
"""clf=svm.SVC(kernel="linear",C=1)
clf.fit(x_train,Y_train)
y_pred=clf.predict(x_test)
print(clf.predict([[5.1,3.5,1.4,0.2]]))
cm=confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True)
plt.xlabel("Iris-setosa  Iris-versicolor  Iris-virginica  \n predicted")
plt.ylabel("actual value")
print(clf.score(x_test,y_test))
plt.show()

# polynomial kernal
clf=svm.SVC(kernel="poly",degree=10,C=1,gamma="auto")
clf.fit(x_train,Y_train)
y_pred=clf.predict(x_test)
print(clf.predict([[7,3.2,4.7,1.4]]))
cm=confusion_matrix(y_test,y_pred)
#sns.heatmap(cm,annot=True)
plt.xlabel("Iris-setosa  Iris-versicolor  Iris-virginica  \n predicted")
plt.ylabel("actual value")
print(clf.score(x_test,y_test))
plt.scatter(df["sepal length (cm)"],df.Target)
plt.show()"""

#Gaussian radial Bias kernal
clf=svm.SVC(kernel="rbf",gamma=0.1,C=1)
clf.fit(x_train,Y_train)
y_pred=clf.predict(x_test)
print(clf.predict([[7,3.2,4.7,1.4]]))
cm=confusion_matrix(y_test,y_pred)
#sns.heatmap(cm,annot=True)
plt.xlabel("Iris-setosa  Iris-versicolor  Iris-virginica  \n predicted")
plt.ylabel("actual value")
print(clf.score(x_test,y_test))
plt.scatter(df["sepal length (cm)"],df.Target)
plt.show()