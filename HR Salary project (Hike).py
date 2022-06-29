import matplotlib.pyplot as plt
import pandas as pd
path="C:\\Users\\HEMANTH\\Downloads\\HR_comma_sep.csv"
df=pd.read_csv(path)
left=df[df.left==1]
df.groupby("left").mean()
#pd.crosstab(df.sales,df.left).plot(kind="bar")
subdf=df[["satisfaction_level","average_montly_hours","promotion_last_5years","left","salary"]]
sal_dummy=pd.get_dummies(subdf.salary)
df_sub=pd.concat([subdf,sal_dummy],axis="columns")
df_sub=df_sub.drop("salary",axis="columns")
x=df_sub.drop("left",axis="columns").values
y=df_sub["left"].values
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.9,random_state=0)
logi=LogisticRegression()
logi.fit(x_train,y_train)
logi.score(x_test,y_test)
y_pred=logi.predict(x_test)
print(logi.predict([[0.11,272,0,0,1,0]]))
from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_test,y_pred)
acu=accuracy_score(y_test,y_pred,normalize=True)
print(acu) # prints the testing accuracy
print(cm)
print(logi.score(x_train,y_train))  #prints the training accuracy
print(logi.score(x_test,y_test))  #prints the testing accuracy
import seaborn as sns
sns.heatmap(cm,annot=True)
plt.xlabel("predicted")
plt.ylabel("truth")
plt.show()