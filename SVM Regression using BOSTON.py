#performed Regression using SVM using Boston dataset where we got highest accuracy score
#in rbf kernal and 2nd highest in default , least in polynomial kernal
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as sd
import pandas as pd
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
data=sd.load_boston()
df=pd.DataFrame(data=data.data,columns=data.feature_names)
y=data.target
x=df[["LSTAT"]].values
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=0)
reg=SVR(kernel="poly",degree=3,C=1e3)
reg.fit(x_train,y_train)
y_pred=reg.predict(x_train)
y_test_pred=reg.predict(x_test)
print(reg.score(x_test,y_test))
print(mean_squared_error(y_train,y_pred))
print(mean_squared_error(y_test,y_test_pred))
print(r2_score(y_train,y_pred))
print(r2_score(y_test,y_test_pred))