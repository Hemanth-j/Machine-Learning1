import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
data=load_iris()
df=pd.DataFrame(data.data,columns=data.feature_names)
df["target"]=data.target
"""print(df)
print(data.data[:,0])"""
"""plt.scatter(data.data[:,0],data.data[:,1],c=data.target,cmap="rainbow")
plt.show()"""
x=df.drop("target",axis="columns")
y=df["target"]
"""from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,train_size=0.8)
#LDA used as a classifier
lda=LinearDiscriminantAnalysis()
lda.fit(x_train,y_train)
print(lda.score(x_test,y_test))
y_pred=lda.predict(x_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
import seaborn as sns
sns.heatmap(cm,annot=True)"""
#plt.show()
#lda used as new features
ldaa=LinearDiscriminantAnalysis(n_components=2)
x_lda=ldaa.fit_transform(x,y)
plt.subplot(1,2,1)
plt.scatter(x_lda[:,0],x_lda[:,1],c=y,cmap="rainbow")
plt.subplot(1,2,2)
plt.scatter(data.data[:,0],data.data[:,1],c=y,cmap="rainbow")
plt.show()