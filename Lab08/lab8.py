#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import datasets
data_breast_cancer = datasets.load_breast_cancer()

from sklearn.datasets import load_iris
data_iris = load_iris()


# In[2]:


from sklearn.decomposition import PCA

X_cancer = data_breast_cancer["data"]
X_iris = data_iris["data"]

pca_cancer = PCA(n_components=0.9)
X2D_cancer = pca_cancer.fit_transform(X_cancer)
print(pca_cancer.explained_variance_ratio_)

pca_iris = PCA(n_components=0.9)
X2D_iris = pca_iris.fit_transform(X_iris)
print(pca_iris.explained_variance_ratio_)


# In[3]:


from sklearn.preprocessing import StandardScaler

scaler_cancer = StandardScaler()
X_cancer_scaler = scaler_cancer.fit_transform(X_cancer)

scaler_iris = StandardScaler()
X_iris_scaler = scaler_iris.fit_transform(X_iris)

pca_cancer_scal = PCA(n_components=0.9)
X2D_cancer_scal = pca_cancer_scal.fit_transform(X_cancer_scaler)
print(pca_cancer_scal.explained_variance_ratio_)

pca_iris_scal = PCA(n_components=0.9)
X2D_iris_scal = pca_iris_scal.fit_transform(X_iris_scaler)
print(pca_iris_scal.explained_variance_ratio_)
c = list(pca_cancer_scal.explained_variance_ratio_)
i = list(pca_iris_scal.explained_variance_ratio_)


# In[4]:


import pickle
with open('pca_bc.pkl', 'wb') as fp:
    pickle.dump(c, fp)
with open('pca_ir.pkl', 'wb') as fp:
    pickle.dump(i, fp)


# In[5]:


import numpy as np
res_cancer = []
for i in pca_cancer_scal.components_:
    res_cancer.append(np.abs(i).argmax())
print(res_cancer)
res_iris = []
for i in pca_iris_scal.components_:
    res_iris.append(np.abs(i).argmax())
print(res_iris)
    


# In[6]:


with open('idx_bc.pkl', 'wb') as fp:
    pickle.dump(res_cancer, fp)
with open('idx_ir.pkl', 'wb') as fp:
    pickle.dump(res_cancer, fp)

