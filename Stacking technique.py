#Stacking illustration
import warnings
warnings.filterwarnings('ignore')
"""import matplotlib.pyplot as plt
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
x=x.join(cat_df)"""
from sklearn.datasets import load_breast_cancer
data=load_breast_cancer()
x=data.data
y=data.target
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=0)
from sklearn.ensemble import GradientBoostingClassifier
g_clf=GradientBoostingClassifier(n_estimators=100,learning_rate=0.1,max_depth=3,random_state=0)
from xgboost import XGBClassifier
x_clf=XGBClassifier(random_state=0)
from lightgbm import LGBMClassifier
l_clf=LGBMClassifier(n_estimators=100,learning_rate=0.1,random_state=0)
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
lo_clf=LogisticRegression()
from sklearn.ensemble import RandomForestClassifier
r_clf=RandomForestClassifier(n_estimators=200,criterion="entropy",max_depth=20)
from sklearn.ensemble import AdaBoostClassifier
a_clf=AdaBoostClassifier(n_estimators=100,learning_rate=0.4,random_state=0)
estimator=[("x_clf",x_clf),("l_clf",l_clf),("g_clf",g_clf),("r_clf",r_clf)]
s_clf=StackingClassifier(estimators=estimator,final_estimator=a_clf)
s_clf.fit(x_train,y_train)
print(s_clf.score(x_test,y_test))