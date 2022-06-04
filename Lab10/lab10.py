#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data() 
assert X_train.shape == (60000, 28, 28)
assert X_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)


# In[2]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)


# In[3]:


import matplotlib.pyplot as plt 
plt.imshow(X_train[142], cmap="binary") 
plt.axis('off')
plt.show()


# In[4]:


class_names = ["koszulka", "spodnie", "pulower", "sukienka", "kurtka",
               "sandał", "koszula", "but", "torba", "kozak"]
class_names[y_train[142]]


# In[5]:


import keras
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])


# In[6]:


model.summary()
#tf.keras.utils.plot_model(model, "fashion_mnist.png", show_shapes=True)


# In[7]:


import os
root_logdir = os.path.join(os.curdir, "image_logs")
def get_run_logdir(): 
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S") 
    return os.path.join(root_logdir, run_id)
run_logdir = get_run_logdir()
tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)


# In[8]:


history = model.fit(X_train, y_train, epochs=20, validation_split=0.1, callbacks=[tensorboard_cb])


# In[9]:


import numpy as np

image_index = np.random.randint(len(X_test))
image = np.array([X_test[image_index]])
confidences = model.predict(image)
confidence = np.max(confidences[0])
prediction = np.argmax(confidences[0])
print("Prediction:", class_names[prediction])
print("Confidence:", confidence)
print("Truth:", class_names[y_test[image_index]])
plt.imshow(image[0], cmap="binary")
plt.axis('off')
plt.show()


# In[10]:


model.save('fashion_clf.h5')


# In[11]:


from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
housing = fetch_california_housing()


# In[12]:


X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, random_state=42)


# In[13]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)


# In[14]:


model_housing1 = keras.models.Sequential([
    keras.layers.Dense(30, activation="relu", input_shape=X_train.shape[1:]),
    keras.layers.Dense(1)
])
model_housing1.compile(loss="mean_squared_error", optimizer='SGD')

model_housing1.summary()


# In[15]:


es = tf.keras.callbacks.EarlyStopping(patience=5,
                                      min_delta=0.01, 
                                      verbose=1)


# In[16]:


import os
root_logdir = os.path.join(os.curdir, "housing_logs")
def get_run_logdir(): 
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S") 
    return os.path.join(root_logdir, run_id)
run_logdir = get_run_logdir()
tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)


# In[17]:


history = model_housing1.fit(X_train, y_train, epochs=100, validation_data=(X_valid, y_valid), callbacks=[tensorboard_cb, es])


# In[18]:


model_housing1.save("reg_housing_1.h5")


# In[19]:


model_housing2 = keras.models.Sequential([
    keras.layers.Dense(90, activation="relu", input_shape=X_train.shape[1:]),
    keras.layers.Dense(90, activation="relu"),
    keras.layers.Dense(90, activation="relu"),
    keras.layers.Dense(1)
])


# In[20]:


model_housing2.compile(loss="mean_squared_error", optimizer='SGD')

model_housing2.summary()


# In[21]:


history = model_housing2.fit(X_train, y_train, epochs=100, validation_data=(X_valid, y_valid), callbacks=[tensorboard_cb, es])


# In[22]:


model_housing2.save("reg_housing_2.h5")


# In[23]:


model_housing3 = keras.models.Sequential([
    keras.layers.Dense(80, activation="relu", input_shape=X_train.shape[1:]),
    keras.layers.Dense(40, activation="relu"),
    keras.layers.Dense(20, activation="relu"),
    keras.layers.Dense(10, activation="relu"),
    keras.layers.Dense(1)
])


# In[24]:


model_housing3.compile(loss="mean_squared_error", optimizer='SGD')

model_housing3.summary()


# In[25]:


history = model_housing3.fit(X_train, y_train, epochs=100, validation_data=(X_valid, y_valid), callbacks=[tensorboard_cb, es])


# In[26]:


model_housing3.save("reg_housing_3.h5")

