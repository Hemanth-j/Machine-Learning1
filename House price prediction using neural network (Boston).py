#regression using neuralnetwork
import pandas as pd
from sklearn.datasets import load_boston
data=load_boston()
x=data.data
X=pd.DataFrame(x)
y=data.target
Y=pd.DataFrame(y)
"""print(x)
print(y)"""
import tensorflow as tf
from tensorflow import keras
from keras import layers as ks
from sklearn.model_selection import train_test_split
from keras import Sequential
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=0)
model=Sequential([ks.Dense(units=100,activation="relu",input_shape=(13,)),
                  ks.Dense(units=50,activation="relu"),
                  ks.Dense(units=10,activation="sigmoid")])
model.compile(optimizer="adam",loss="mean_squared_error",metrics=["accuracy"])
histroy=model.fit(x_train,y_train,epochs=500)
#model.save("my model")
print(histroy)
