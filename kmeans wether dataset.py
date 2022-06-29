import matplotlib.pyplot as plt
import pandas as pd
df=pd.read_csv("C:\\Users\\HEMANTH\\Downloads\\archive\\minute_weather.csv")
print(df.shape)
df1=df.dropna()
#print(df1.shape)
#print(df1.isna().sum())
#print(df1.describe())
df1=df1.drop("rowID",axis="columns")
df1=df1.drop("hpwren_timestamp",axis="columns")
#print(df1["rain_duration"].describe())
df1=df1.drop("rain_accumulation",axis="columns")
df1=df1.drop("rain_duration",axis="columns")
#print(df1.columns)
from sklearn.preprocessing import StandardScaler
s_scl=StandardScaler()
x_scl=s_scl.fit_transform(df1)
#print(x_scl)
from sklearn.cluster import KMeans
arr=[]
"""for i in range(1,15):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(x_scl)
    arr.append(kmeans.inertia_)
print(arr)"""
y=[14281407.000000153, 11000196.221698858, 8818644.006892493, 7655696.041691813, 6726527.8772164155, 6026916.4647878045]
x=[]
for i in range(1,7):
    x.append(i)
#print(x)
"""import matplotlib.pyplot as plt
plt.plot(x,y)
plt.show()"""
kmeans = KMeans(n_clusters=3)
y_pred=kmeans.fit_predict(x_scl)
from mlxtend.plotting import plot_decision_regions
plot_decision_regions(x_scl,y_pred,kmeans)
plt.show()