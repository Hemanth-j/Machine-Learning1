#ransac robust regression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.metrics
from sklearn.linear_model import RANSACRegressor
import seaborn as sns
from sklearn.metrics import r2_score
df=pd.read_csv("C:\\Users\\HEMANTH\\Downloads\\archive\\income.data_\\income.data.csv")
x=np.array(df.income).reshape(-1,1)
y=np.array(df.happiness).reshape(-1,1)
ranasc=RANSACRegressor()
ranasc.fit(x,y)
inlier_mask=ranasc.inlier_mask_
outlier_mask=np.logical_not(inlier_mask)
line_x=np.arange(3,10,1)
line_y_ransac=ranasc.predict(np.array(line_x).reshape(-1,1))
sns.set(style="darkgrid",context="notebook")
plt.scatter(x[inlier_mask],y[inlier_mask],color="blue",marker="+")
plt.scatter(x[outlier_mask],y[outlier_mask],color="yellow",marker="s")
plt.plot(line_x,line_y_ransac,color="red")
plt.xlabel("INCOME")
plt.ylabel("HAPPINESS")
plt.legend("upper right")
plt.title("ROBUST REGRESSION")
print(ranasc.score(x,y))
print(ranasc.predict([[3.86264741839841]]))
print(sklearn.metrics.mean_squared_error(y,ranasc.predict(np.array(x).reshape(-1,1))))
print(r2_score(y,ranasc.predict(np.array(x).reshape(-1,1))))
plt.show()