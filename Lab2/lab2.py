#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1)


# In[ ]:


print(mnist.target)
print(mnist.data)


# In[ ]:


print((np.array(mnist.data.loc[42]).reshape(28, 28) > 0).astype(int))


# In[ ]:


X = pd.DataFrame(mnist.data)
y = pd.DataFrame(mnist.target)
y = y.sort_values(by='class')
print(y)

X = X.reindex(y.index)
#X = X.reindex_like(other=y)
print(X.index)


# In[ ]:


X_train, X_test = X[:56000], X[56000:]
y_train, y_test = y[:56000], y[56000:]
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# In[ ]:


pd.unique(y_train['class'])


# In[ ]:


pd.unique(y_test['class'])


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[ ]:


pd.unique(y_train['class'])


# In[ ]:


pd.unique(y_test['class'])


# In[ ]:


from sklearn.linear_model import SGDClassifier
y_train_0 = (y_train == '0')
y_test_0 = (y_test == '0')


# In[ ]:


sgd_clf = SGDClassifier()
sgd_clf.fit(X_train, y_train_0)


# In[ ]:


y_pred_test = sgd_clf.predict(X_test)
y_pred_train = sgd_clf.predict(X_train)


# In[ ]:


from sklearn.metrics import accuracy_score
results = list()
results.append(accuracy_score(y_train_0, y_pred_train))
results.append(accuracy_score(y_test_0, y_pred_test))

print(results)


# In[ ]:


import pickle
with open('sgd_acc.pkl', 'wb') as f:
    pickle.dump(results, f)


# In[ ]:


from sklearn.model_selection import cross_val_score
score = cross_val_score(sgd_clf, X_train, y_train_0,
                        cv=3, scoring="accuracy",
                        n_jobs=-1)
print(type(score))


# In[ ]:


with open('sgd_cva.pkl', 'wb') as f:
    pickle.dump(score, f)


# In[ ]:


sgd_clf = SGDClassifier()
sgd_clf.fit(X_train, y_train)


# In[25]:


from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train, cv=3, n_jobs=-1)
print(y_train_pred)
print(len(y_train_pred))
cmx = confusion_matrix(y_train, y_train_pred)
print(cmx)


# In[28]:


with open('sgd_cmx.pkl', 'wb') as f:
    pickle.dump(cmx, f)


# In[27]:





# In[ ]:




