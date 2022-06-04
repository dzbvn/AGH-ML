#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import datasets
import numpy as np

data_breast_cancer = datasets.load_breast_cancer(as_frame=True)
#print(data_breast_cancer['data'][['mean area', 'mean smoothness']])
X_cancer = data_breast_cancer['data'][['mean area', 'mean smoothness']]
y_cancer = data_breast_cancer['target']


# In[2]:


data_iris = datasets.load_iris(as_frame=True)
X_iris = data_iris['data'][['petal length (cm)', 'petal width (cm)']]
y_iris = (data_iris["target"] == 2).astype(np.int8)


# In[3]:


from sklearn.model_selection import train_test_split
X_cancer_train, X_cancer_test, y_cancer_train, y_cancer_test = train_test_split(X_cancer, y_cancer, test_size=0.2)
X_iris_train, X_iris_test, y_iris_train, y_iris_test = train_test_split(X_iris, y_iris, test_size=0.2)


# In[4]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

svm_clf_cancer = Pipeline([
        ("linear_svc", LinearSVC(C=1,
                                 loss="hinge",
                                ))])
svm_clf_cancer.fit(X_cancer_train, y_cancer_train)


# In[5]:


svm_clf_cancer_scal = Pipeline([
        ("scaler", StandardScaler()),
        ("linear_svc", LinearSVC(C=1,
                                 loss="hinge",
                                ))])
svm_clf_cancer_scal.fit(X_cancer_train, y_cancer_train)


# In[6]:


from sklearn.metrics import accuracy_score

#zbiór uczący bez skalowania
y_cancer_train_pred = svm_clf_cancer.predict(X_cancer_train)
acc_cancer_train = accuracy_score(y_cancer_train, y_cancer_train_pred)
print(acc_cancer_train)

#zbiór testujący bez skalowania
y_cancer_test_pred = svm_clf_cancer.predict(X_cancer_test)
acc_cancer_test = accuracy_score(y_cancer_test, y_cancer_test_pred)
print(acc_cancer_test)

#zbiór uczący ze skalowaniem
y_cancer_train_pred_scal = svm_clf_cancer_scal.predict(X_cancer_train)
acc_cancer_train_scal = accuracy_score(y_cancer_train, y_cancer_train_pred_scal)
print(acc_cancer_train_scal)

#zbiór testujący ze skalowaniem
y_cancer_test_pred_scal = svm_clf_cancer_scal.predict(X_cancer_test)
acc_cancer_test_scal = accuracy_score(y_cancer_test, y_cancer_test_pred_scal)
print(acc_cancer_test_scal)


# In[7]:


bc_acc = [acc_cancer_train, acc_cancer_test, acc_cancer_train_scal, acc_cancer_test_scal]
print(bc_acc)


# In[8]:


import pickle
with open('bc_acc.pkl', 'wb') as fp:
    pickle.dump(bc_acc, fp)


# In[9]:


svm_clf_iris = Pipeline([
        ("linear_svc", LinearSVC(C=1,
                                 loss="hinge",
                                ))])
svm_clf_iris.fit(X_iris_train, y_iris_train)


# In[10]:


svm_clf_iris_scal = Pipeline([
        ("scaler", StandardScaler()),
        ("linear_svc", LinearSVC(C=1,
                                 loss="hinge",
                                ))])
svm_clf_iris_scal.fit(X_iris_train, y_iris_train)


# In[11]:


from sklearn.metrics import accuracy_score

#zbiór uczący bez skalowania
y_iris_train_pred = svm_clf_iris.predict(X_iris_train)
acc_iris_train = accuracy_score(y_iris_train, y_iris_train_pred)
print(acc_iris_train)

#zbiór testujący bez skalowania
y_iris_test_pred = svm_clf_iris.predict(X_iris_test)
acc_iris_test = accuracy_score(y_iris_test, y_iris_test_pred)
print(acc_iris_test)

#zbiór uczący ze skalowaniem
y_iris_train_pred_scal = svm_clf_iris_scal.predict(X_iris_train)
acc_iris_train_scal = accuracy_score(y_iris_train, y_iris_train_pred_scal)
print(acc_iris_train_scal)

#zbiór testujący ze skalowaniem
y_iris_test_pred_scal = svm_clf_iris_scal.predict(X_iris_test)
acc_iris_test_scal = accuracy_score(y_iris_test, y_iris_test_pred_scal)
print(acc_iris_test_scal)


# In[12]:


iris_acc = [acc_iris_train, acc_iris_test, acc_iris_train_scal, acc_iris_test_scal]


# In[13]:


with open('iris_acc.pkl', 'wb') as fp:
    pickle.dump(iris_acc, fp)

