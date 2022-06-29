# IBM HR DataAnlytics Project , using Stacking
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import time
start=time.time()
import pandas as pd
df=pd.read_csv("C:\\Users\\HEMANTH\\Downloads\\WA_Fn-UseC_-HR-Employee-Attrition.csv")
data=df.drop("EmployeeCount",axis="columns")
data=data.drop("EmployeeNumber",axis="columns")
data=data.drop("Over18",axis="columns")
data=data.drop("StandardHours",axis="columns")
y=data["Attrition"].map({"Yes":1,"No":0})
num_df=data.describe().columns
data=data.drop("Attrition",axis="columns")
b=set(data.columns)-set(num_df)
cat_df=pd.get_dummies(df[b])
x=data[num_df]
x=x.join(cat_df)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=0)
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
clf4=LogisticRegression()
from sklearn.ensemble import StackingClassifier
estimators=[("rf",RandomForestClassifier()),("DT",DecisionTreeClassifier()),("SVC",SVC())]
s_clf=StackingClassifier(estimators=estimators,final_estimator=clf4)
s_clf.fit(x_train,y_train)
print(s_clf.score(x_test,y_test))