#!/usr/bin/env python
# coding: utf-8

# ### Zbiór danych

# mkdir data

# In[ ]:


import os
os.mkdir("./data")


# curl -O https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.tgz

# In[ ]:


import urllib.request
filename = "housing.tgz"
url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.tgz"
fullfilename = os.path.join("./data", filename)
urllib.request.urlretrieve(url, fullfilename)


# tar xfz housing.tgz

# In[ ]:


import tarfile
my_tar = tarfile.open('./data/housing.tgz')
my_tar.extractall('./data')
my_tar.close()


# gzip housing.csv

# In[ ]:


import gzip
import shutil
with open('./data/housing.csv', 'rb') as f_in:
    with gzip.open('./data/housing.csv.gz', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)


# rm housing.tgz

# In[ ]:


os.remove('./data/housing.tgz')
os.remove('./data/housing.csv')



# gzcat data/housing.csv.gz | head -4

# In[ ]:


file = gzip.open('./data/housing.csv.gz','rb')
content = file.readline()
print(content)
content = file.readline()
print(content)


# ### Informacje o zbiorze danych

# Użyj pandas aby zobaczyć czym są pobrane dane.
# Zmienna df to DataFrame, mozemy porownac do tabeli.
# Dane w kolumnie 'ocean_proximity' są typu object.

# In[ ]:


import pandas as pd
df = pd.read_csv('data/housing.csv.gz')
#df.head()
df.info()
df['ocean_proximity'].head()
#print(type(df['ocean_proximity'][0]))
#df['ocean_proximity'].value_counts()
#df['ocean_proximity'].describe()


# ### Wizualizacja

# In[ ]:


import matplotlib.pyplot as plt
df.hist(bins=50, figsize=(20,15))
plt.savefig('./data/obraz1.png')


# In[ ]:


df.plot(kind="scatter", x="longitude", y="latitude",
        alpha=0.1, figsize=(7,4))
plt.savefig('./data/obraz2.png')


# In[ ]:


import matplotlib.pyplot as plt    # potrzebne ze względu na argument cmap
df.plot(kind="scatter", x="longitude", y="latitude",
        alpha=0.4, figsize=(7,3), colorbar=True,
        s=df["population"]/100, label="population",
        c="median_house_value", cmap=plt.get_cmap("jet"))
plt.savefig('./data/obraz3.png')


# ### Analiza

# In[ ]:


df.corr()["median_house_value"].sort_values(ascending=False)


# 

# In[ ]:


s = df.corr()["median_house_value"].sort_values(ascending=False).reset_index()
final = s.rename(columns={"index":"atrybut", "median_house_value":"wspolczynnik_korelacji"})
final
final.to_csv('./data/korelacja.csv', index=False)


# Do wizualnej analizy związków pomiędzy zmiennymi często używamy tzw. pair plot. 

# In[ ]:


import seaborn as sns
sns.pairplot(df)


# ### Przygotowanie do uczenia

# In[ ]:


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
len(train_set),len(test_set)
#train_set.head()
#test_set.head()
#train_set.corr()
#test_set.corr()
test_set.corr()["median_house_value"].sort_values(ascending=False)
#train_set.corr()["median_house_value"].sort_values(ascending=False)


# Wyniki są podobne, oznacza to, że na wartość domu czynniki w obu zbiorach maja podobny wpływ

# In[ ]:


import pickle
test_set.to_pickle('./data/test_set.pkl')
train_set.to_pickle('./data/train_set.pkl')


# In[ ]:


import matplotlib.pyplot as plt
pd.options.plotting.backend = "plotly"
df.hist(bins=50, figsize=(20,15))
plt.savefig('./data/obraz1.png')

