#!/usr/bin/env python
# coding: utf-8

# In[110]:


from sklearn.datasets import load_iris
iris = load_iris(as_frame=True)


# In[111]:


import pandas as pd

pd.concat([iris.data, iris.target], axis=1).plot.scatter(
    x='petal length (cm)',
    y='petal width (cm)',
    c='target',
    colormap='viridis'
)


# In[112]:


from sklearn.model_selection import train_test_split

X = iris["data"][["petal length (cm)", "petal width (cm)"]]
y = iris["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[113]:


from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

per_clf_0 = Perceptron()

y_train_0 = (y_train == 0).astype(int)
y_test_0 = (y_test == 0).astype(int)

per_clf_0.fit(X_train, y_train_0)

y_pred_train_0 = per_clf_0.predict(X_train)
y_pred_test_0 = per_clf_0.predict(X_test)

acc_train_0 = accuracy_score(y_train_0, y_pred_train_0)
acc_test_0 = accuracy_score(y_test_0, y_pred_test_0)


# In[114]:


per_clf_1 = Perceptron()

y_train_1 = (y_train == 1).astype(int)
y_test_1 = (y_test == 1).astype(int)

per_clf_1.fit(X_train, y_train_1)

y_pred_train_1 = per_clf_1.predict(X_train)
y_pred_test_1 = per_clf_1.predict(X_test)

acc_train_1 = accuracy_score(y_train_1, y_pred_train_1)
acc_test_1 = accuracy_score(y_test_1, y_pred_test_1)


# In[115]:


per_clf_2 = Perceptron()

y_train_2 = (y_train == 2).astype(int)
y_test_2 = (y_test == 2).astype(int)

per_clf_2.fit(X_train, y_train_2)

y_pred_train_2 = per_clf_2.predict(X_train)
y_pred_test_2 = per_clf_2.predict(X_test)

acc_train_2 = accuracy_score(y_train_2, y_pred_train_2)
acc_test_2 = accuracy_score(y_test_2, y_pred_test_2)


# In[116]:


per_acc = [(acc_train_0, acc_test_0), (acc_train_1, acc_test_1), (acc_train_2, acc_test_2)]
print(per_acc)


# In[117]:


per_wght = []
for p in [per_clf_0, per_clf_1, per_clf_2]:
    per_wght.append((p.intercept_[0], p.coef_[0, 0], p.coef_[0, 1]))

print(per_wght)


# In[118]:


import pickle
with open('per_acc.pkl', 'wb') as fp:
    pickle.dump(per_acc, fp)
with open('per_wght.pkl', 'wb') as fp:
    pickle.dump(per_wght, fp)


# In[119]:


import numpy as np

X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([0,
              1,
              1,
              0])


# In[120]:


import keras
import tensorflow as tf

model = keras.models.Sequential()
model.add(keras.layers.Dense(2, activation="tanh", use_bias=True, input_dim=2))
model.add(keras.layers.Dense(1, activation="sigmoid", use_bias=True))

model.summary()


# In[121]:


model.compile(loss="binary_crossentropy", optimizer="sgd")

history = model.fit(X, y, epochs=100, verbose=False)
print(history.history['loss'])


# In[122]:


model.predict(X)


# In[123]:


counter = 0
while True:
    modelS = keras.models.Sequential()
    modelS.add(keras.layers.Dense(2, activation="tanh", use_bias=True, input_dim=2))
    modelS.add(keras.layers.Dense(1, activation="sigmoid", use_bias=True))
    modelS.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.2))

    history = modelS.fit(X, y, epochs=100, verbose=False)

    res = modelS.predict(X)
    counter += 1
    if res[0] < 0.1 and res[3] < 0.1 and res[1] > 0.9 and res[2] > 0.9:
        break
print(counter, res)


# In[124]:


with open('mlp_xor_weights.pkl', 'wb') as fp:
    pickle.dump(modelS.get_weights(), fp)

