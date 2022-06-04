#!/usr/bin/env python
# coding: utf-8

# In[554]:


import sklearn.neighbors
from sklearn import datasets
data_breast_cancer = datasets.load_breast_cancer(as_frame=True)


# In[555]:


X_cancer = data_breast_cancer['data'][['mean texture', 'mean symmetry']]
y_cancer = data_breast_cancer['target']


# In[556]:


from sklearn.model_selection import train_test_split
X_cancer_train, X_cancer_test, y_cancer_train, y_cancer_test = train_test_split(X_cancer, y_cancer, test_size=0.2, random_state=21)


# In[557]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier

tree_clf = DecisionTreeClassifier()

log_clf = LogisticRegression()

knn_clf = sklearn.neighbors.KNeighborsClassifier()

voting_hard_clf = VotingClassifier(
    estimators=[('dt', tree_clf),
                ('lr', log_clf),
                ('knn', knn_clf)],
    voting='hard')

voting_soft_clf = VotingClassifier(
    estimators=[('dt', tree_clf),
                ('lr', log_clf),
                ('knn', knn_clf)],
    voting='soft')

voting_hard_clf.fit(X_cancer_train, y_cancer_train)
voting_soft_clf.fit(X_cancer_train, y_cancer_train)


# In[558]:


voting_hard_clf.predict(X_cancer_test)
voting_soft_clf.predict(X_cancer_test)


# In[559]:


from sklearn.metrics import accuracy_score

acc_list = list()

for clf in (tree_clf, log_clf, knn_clf, voting_hard_clf, voting_soft_clf):
    clf.fit(X_cancer_train, y_cancer_train)

    y_pred_train = clf.predict(X_cancer_train)
    acc_train = accuracy_score(y_cancer_train, y_pred_train)

    y_pred_test = clf.predict(X_cancer_test)
    acc_test = accuracy_score(y_cancer_test, y_pred_test)

    acc_list.append((acc_train, acc_test))

print(acc_list)


# In[560]:


clf_list = [tree_clf, log_clf, knn_clf, voting_hard_clf, voting_soft_clf]
print(clf_list)


# In[561]:


import pickle
with open('acc_vote.pkl', 'wb') as fp:
    pickle.dump(acc_list, fp)

with open('vote.pkl', 'wb') as fp:
    pickle.dump(clf_list, fp)


# In[562]:


from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

#bagging
bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=30, bootstrap=True, random_state=42)

#bagging 1/2
bag_half_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=30, max_samples=0.5, bootstrap=True, random_state=42)

#pasting
pas_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=30, bootstrap=False, random_state=42)

#pasting 1/2
pas_half_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=30, max_samples=0.5, bootstrap=False, random_state=42)

#random forest
rnd_clf = RandomForestClassifier(n_estimators=30, random_state=42)

#ada boost
ada_clf = AdaBoostClassifier(n_estimators=30, random_state=42)

#gradient boost
gdt_clf = GradientBoostingClassifier(n_estimators=30, random_state=42)


# In[563]:


acc_bag = list()
for clf in (bag_clf, bag_half_clf, pas_clf, pas_half_clf, rnd_clf, ada_clf, gdt_clf):
    clf.fit(X_cancer_train, y_cancer_train)

    y_pred_train = clf.predict(X_cancer_train)
    acc_train = accuracy_score(y_cancer_train, y_pred_train)

    y_pred_test = clf.predict(X_cancer_test)
    acc_test = accuracy_score(y_cancer_test, y_pred_test)

    acc_bag.append((acc_train, acc_test))
print(acc_bag)


# In[564]:


bag_list = [bag_clf, bag_half_clf, pas_clf, pas_half_clf, rnd_clf, ada_clf, gdt_clf]


# In[565]:


with open('acc_bag.pkl', 'wb') as fp:
    pickle.dump(acc_bag, fp)

with open('bag.pkl', 'wb') as fp:
    pickle.dump(bag_list, fp)


# In[566]:


X = data_breast_cancer['data']
y = data_breast_cancer['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)


# In[567]:


clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=30, max_samples=0.5, max_features=2, bootstrap_features=False, bootstrap=True, random_state=42)
clf.fit(X_train, y_train)

y_pred_train = clf.predict(X_train)
acc_train = accuracy_score(y_train, y_pred_train)

y_pred_test = clf.predict(X_test)
acc_test = accuracy_score(y_test, y_pred_test)

acc_fea = [acc_train, acc_test]
fea = [clf]
print(acc_fea)


# In[568]:


with open('acc_fea.pkl', 'wb') as fp:
    pickle.dump(acc_fea, fp)

with open('fea.pkl', 'wb') as fp:
    pickle.dump(fea, fp)


# In[569]:


res = list()

for estimator, features in zip(clf.estimators_, clf.estimators_features_):
    #print(estimator, features)

    y_pred_train = estimator.predict(X_train.iloc[:, features].values)
    acc_train = accuracy_score(y_train, y_pred_train)

    y_pred_test = estimator.predict(X_test.iloc[:, features].values)
    acc_test = accuracy_score(y_test, y_pred_test)

    res.append([acc_train, acc_test, list(X.columns[features])])


# In[570]:


import pandas as pd
#print(res)
res = pd.DataFrame(res, columns=['Train accuracy', 'Test accuracy', 'Features'])
res.sort_values(['Test accuracy', 'Train accuracy'], ascending=False, inplace=True)
print(res)


# In[571]:


with open('acc_fea_rank.pkl', 'wb') as fp:
    pickle.dump(res, fp)

