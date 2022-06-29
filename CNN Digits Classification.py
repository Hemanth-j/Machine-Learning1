#digit classification using Neural Network MNIST
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import keras.layers as ks
import numpy as np
(x_train,y_train),(x_test,y_test)=keras.datasets.mnist.load_data()
x_trainn=x_train/255
x_testt=x_test/255
print(y_train)
print(y_test)
x_train_flattened=np.array(x_trainn).reshape(len(x_trainn),28*28)
x_test_flattened=np.array(x_testt).reshape(len(x_testt),28*28)
model=keras.Sequential([
   ks.Dense(100,input_shape=(784,),activation="relu"),
   ks.Dense(50,activation="relu"),
   ks.Dense(10,activation="sigmoid")
])
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
c=model.fit(x_train_flattened,y_train,epochs=5)
model.evaluate(x_test_flattened,y_test)
plt.matshow(x_test[3])
y_predicted=model.predict(x_test_flattened)
print(np.argmax(y_predicted[3]))
plt.show()