#!/usr/bin/env python
# coding: utf-8

# In[277]:


import numpy as np
import pandas as pd

size = 300
X = np.random.rand(size)*5-2.5
w4, w3, w2, w1, w0 = 1, 2, 1, -4, 2
y = w4*(X**4) + w3*(X**3) + w2*(X**2) + w1*X + w0 + np.random.randn(size)*8-4
df = pd.DataFrame({'x': X, 'y': y})
df.to_csv('dane_do_regresji.csv',index=None)
df.plot.scatter(x='x',y='y')
#X = df['x']
#y = df['y']
#print(X.shape)
#print(df['x'])


# In[278]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#print(X_test.shape, X_train.shape)
#print(X_train)
#print(X_train)

X_train = X_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)


# In[279]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

lin_reg = LinearRegression()

lin_reg.fit(X_train, y_train)
X_new = np.array([[0], [2]])
print(X_new)

print(lin_reg.intercept_, lin_reg.coef_, "\n",
lin_reg.predict(X_new))

lin_train_mse = mean_squared_error(y_train, lin_reg.predict(X_train))
lin_test_mse = mean_squared_error(y_test, lin_reg.predict(X_test))
print(lin_train_mse, lin_test_mse)


# In[280]:


import sklearn.neighbors

knn_3_reg = sklearn.neighbors.KNeighborsRegressor(
n_neighbors=3)
knn_3_reg.fit(X_train, y_train)

knn3_train_mse = mean_squared_error(y_train, knn_3_reg.predict(X_train))
knn3_test_mse = mean_squared_error(y_test, knn_3_reg.predict(X_test))
print(knn3_train_mse, knn3_test_mse)


# In[281]:


import sklearn.neighbors

knn_5_reg = sklearn.neighbors.KNeighborsRegressor(
n_neighbors=5)
knn_5_reg.fit(X_train, y_train)

knn5_train_mse = mean_squared_error(y_train, knn_5_reg.predict(X_train))
knn5_test_mse = mean_squared_error(y_test, knn_5_reg.predict(X_test))
print(knn5_train_mse, knn5_test_mse)


# In[282]:


from sklearn.preprocessing import PolynomialFeatures
poly_feature_2 = PolynomialFeatures(degree=2, include_bias=False)

X_poly_train = poly_feature_2.fit_transform(X_train)
X_poly_test = poly_feature_2.fit_transform(X_test)

#print(X_train[0], X_poly_train[0])
poly_2_reg = LinearRegression()
poly_2_reg.fit(X_poly_train, y_train)
print(poly_2_reg.intercept_, poly_2_reg.coef_)
print(poly_2_reg.predict(poly_feature_2.fit_transform([[0],[2]])))
print(poly_2_reg.coef_[0][1] * 2**2 + poly_2_reg.coef_[0][0] * 2 + poly_2_reg.intercept_[0])

poly2_train_mse = mean_squared_error(y_train, poly_2_reg.predict(X_poly_train))
poly2_test_mse = mean_squared_error(y_test, poly_2_reg.predict(X_poly_test))
print(poly2_train_mse, poly2_test_mse)

import matplotlib.pyplot as plt

X_sort = pd.Series(X_test.flatten()).sort_values()
y_sort = pd.Series((poly_2_reg.predict(X_poly_test)).flatten()).reindex(X_sort.index)

plt.plot(X_sort, y_sort)


# In[283]:


from sklearn.preprocessing import PolynomialFeatures
poly_feature_3 = PolynomialFeatures(degree=3, include_bias=False)
X_poly_train = poly_feature_3.fit_transform(X_train)
X_poly_test = poly_feature_3.fit_transform(X_test)
#print(X_train[0], X_poly_train[0])
poly_3_reg = LinearRegression()
poly_3_reg.fit(X_poly_train, y_train)
print(poly_3_reg.intercept_, poly_3_reg.coef_)
print(poly_3_reg.predict(poly_feature_3.fit_transform([[0],[2]])))
print(poly_3_reg.coef_[0][1] * 2**2 + poly_3_reg.coef_[0][0] * 2 + poly_3_reg.intercept_[0])

poly3_train_mse = mean_squared_error(y_train, poly_3_reg.predict(X_poly_train))
poly3_test_mse = mean_squared_error(y_test, poly_3_reg.predict(X_poly_test))
print(poly3_train_mse, poly3_test_mse)

import matplotlib.pyplot as plt
X_sort = pd.Series(X_test.flatten()).sort_values()
y_sort = pd.Series((poly_3_reg.predict(X_poly_test)).flatten()).reindex(X_sort.index)
plt.plot(X_sort, y_sort)


# In[284]:


from sklearn.preprocessing import PolynomialFeatures
poly_feature_4 = PolynomialFeatures(degree=4, include_bias=False)
X_poly_train = poly_feature_4.fit_transform(X_train)
X_poly_test = poly_feature_4.fit_transform(X_test)
#print(X_train[0], X_poly_train[0])
poly_4_reg = LinearRegression()
poly_4_reg.fit(X_poly_train, y_train)
print(poly_4_reg.intercept_, poly_4_reg.coef_)
print(poly_4_reg.predict(poly_feature_4.fit_transform([[0],[2]])))
print(poly_4_reg.coef_[0][1] * 2**2 + poly_4_reg.coef_[0][0] * 2 + poly_4_reg.intercept_[0])

poly4_train_mse = mean_squared_error(y_train, poly_4_reg.predict(X_poly_train))
poly4_test_mse = mean_squared_error(y_test, poly_4_reg.predict(X_poly_test))
print(poly4_train_mse, poly4_test_mse)

import matplotlib.pyplot as plt
X_sort = pd.Series(X_test.flatten()).sort_values()
y_sort = pd.Series((poly_4_reg.predict(X_poly_test)).flatten()).reindex(X_sort.index)
plt.plot(X_sort, y_sort)


# In[285]:


from sklearn.preprocessing import PolynomialFeatures
poly_feature_5 = PolynomialFeatures(degree=5, include_bias=False)
X_poly_train = poly_feature_5.fit_transform(X_train)
X_poly_test = poly_feature_5.fit_transform(X_test)
print(X_train[0], X_poly_train[0])
poly_5_reg = LinearRegression()
poly_5_reg.fit(X_poly_train, y_train)
print(poly_5_reg.intercept_, poly_5_reg.coef_)
print(poly_5_reg.predict(poly_feature_5.fit_transform([[0],[2]])))
print(poly_5_reg.coef_[0][1] * 2**2 + poly_5_reg.coef_[0][0] * 2 + poly_5_reg.intercept_[0])

poly5_train_mse = mean_squared_error(y_train, poly_5_reg.predict((X_poly_train)))
poly5_test_mse = mean_squared_error(y_test, poly_5_reg.predict((X_poly_test)))
print(poly5_train_mse, poly5_test_mse)
import matplotlib.pyplot as plt
X_sort = pd.Series(X_test.flatten()).sort_values()
y_sort = pd.Series((poly_5_reg.predict(X_poly_test)).flatten()).reindex(X_sort.index)
plt.plot(X_sort, y_sort)


# In[286]:




mse_list = [[lin_train_mse, lin_test_mse], [knn3_train_mse, knn3_test_mse], [knn5_test_mse, knn5_test_mse], [poly2_train_mse, poly2_test_mse], [poly3_train_mse, poly3_test_mse], [poly4_train_mse, poly4_test_mse], [poly5_train_mse, poly5_test_mse]]

mse_df = pd.DataFrame(mse_list, index=["lin_reg", "knn_3_reg", "knn_5_reg", "poly_2_reg", "poly_3_reg", "poly_4_reg", "poly_5_reg"], columns=["train_mse", "test_mse"])
print(mse_df)
import pickle
with open('mse.pkl', 'wb') as fp:
    pickle.dump(mse_df, fp)


# In[287]:


reg = list()
reg.append((lin_reg, None))
reg.append((knn_3_reg, None))
reg.append((knn_5_reg, None))
reg.append((poly_2_reg, poly_feature_2))
reg.append((poly_3_reg, poly_feature_3))
reg.append((poly_4_reg, poly_feature_4))
reg.append((poly_5_reg, poly_feature_5))
print(reg)
with open('reg.pkl', 'wb') as fp:
    pickle.dump(reg, fp)

