#explains the difference between PCA and Kernel PCA
from sklearn.decomposition import KernelPCA,PCA
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import pandas as pd
data=load_iris()
df=pd.DataFrame(data.data,columns=data.feature_names)
x=df
y=data.target
from sklearn.preprocessing import StandardScaler
s_scl=StandardScaler()
x_scl=s_scl.fit_transform(x)
pca=PCA()
x_pca=pca.fit_transform(x)
k_pca=KernelPCA(kernel="linear",gamma=5,n_components=3)
K_pca=k_pca.fit_transform(x_scl)
plt.title("Dimensionality reduction")
plt.subplot(1,2,1)
plt.scatter(x_pca[:,0],x_pca[:,1],c=y)
plt.subplot(1,2,2)
plt.scatter(K_pca[:,0],K_pca[:,1],c=y)
plt.show()