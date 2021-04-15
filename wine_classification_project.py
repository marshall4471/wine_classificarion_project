#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
                    


# In[2]:


import numpy as np


# In[3]:


df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)


# In[4]:


df_wine.columns = ['Class label', 'Alcohol',
                 'Malic acid', 'Ash',
                 'Alcalinitity of ash', 'Magnesium',
                 'Total phenols', 'Falvanoids',
                 'Nonflavanoid phenols',
                   'Profoanthocyanins',
                 'Color intensity', 'Hue',
                 'OD280/OD315 of diluted wines',
                 'Proline']


# In[5]:


print('Class labels', np.unique(df_wine['Class label']))


# In[6]:


df_wine.head()


# In[7]:


from sklearn.model_selection import train_test_split


# In[8]:


X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values


# In[9]:


X_train, X_test, y_train, y_test =        train_test_split(X, y,
                        test_size=0.3,
                        random_state=0,
                        stratify=y)


# In[10]:


from sklearn.preprocessing import MinMaxScaler


# In[11]:


mms = MinMaxScaler()


# In[12]:


X_train_norm = mms.fit_transform(X_train)


# In[13]:


X_test_norm = mms.transform(X_test)


# In[14]:


ex = np.array([0, 1, 2, 3, 4, 5])


# In[15]:


print('standardized:', (ex - ex.mean()) / ex.std())


# In[16]:


print('normalized:', (ex - ex.min()) / (ex.max() - ex.min()))


# In[17]:


from sklearn.preprocessing import StandardScaler


# In[18]:


stdsc = StandardScaler()


# In[19]:


X_train_std = stdsc.fit_transform(X_train)


# In[20]:


X_test_std = stdsc.transform(X_test)


# In[21]:


from sklearn.linear_model import LogisticRegression


# In[22]:


LogisticRegression(penalty='l1',
                  solver='liblinear',
                  multi_class='ovr')


# In[23]:


lr = LogisticRegression(penalty='l1',
                       C=1.0,
                       solver='liblinear',
                       multi_class='ovr')


# In[24]:


lr.fit(X_train_std, y_train)


# In[25]:


print('Training accuracy:', lr.score(X_test_std, y_test))


# In[26]:


lr.intercept_


# In[27]:


lr.coef_


# In[28]:


import matplotlib.pyplot as plt


# In[29]:


fig = plt.figure()
ax = plt.subplot(111)
colors =  ['blue', 'green', 'red', 'cyan',
          'magenta', 'yellow', 'black',
          'pink', 'lightgreen', 'lightblue',
          'gray', 'indigo', 'orange']
weights, params = [], []
for c in np.arange(-4., 6.):
    lr = LogisticRegression(penalty='l1', C=10.**c,
                           solver='liblinear',
                           multi_class='ovr', random_state=0)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10**c)
weights = np.array(weights)
for column, color in zip(range(weights.shape[1]), colors):
    plt.plot(params, weights[:, column],
            label=df_wine.columns[column + 1],
            color=color)
plt.axhline(0, color='black', linestyle='--', linewidth=3)
plt.xlim([10**(-5), 10**5])
plt.ylabel('weight coefficient')
plt.xlabel('C')
plt.xscale('log')
plt.legend(loc='upper left')
ax.legend(loc='upper center',
         bbox_to_anchor=(1.38, 1.03),
         ncol=1, fancybox=True)
plt.show()


# In[30]:


from sklearn.base import clone


# In[31]:


from itertools import combinations


# In[32]:


import numpy as py


# In[33]:


from sklearn.metrics import accuracy_score


# In[34]:


from sklearn.model_selection import train_test_split


# In[71]:


class SBS():
    def __init__(self, estimator, k_features,
             scoring=accuracy_score,
             test_size=0.25, random_state=1):
    
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state
        
    def fit(self, X, y):
        X_train, X_test, y_train, y_test =         train_test_split(X, y, test_size=self.test_size,
                        random_state=self.random_state)
        dim= X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        score = self._calc_score(X_train, y_train, X_test, y_test, self.indices_)
        
        self.scores_ = [score]
        
        while dim > self.k_features:
            scores = []
            subsets = []
            
            for p in combinations(self.indices_, r=dim -1):
                score = self._calc_score(X_train, y_train, 
                                        X_test, y_test, p)
                scores.append(score)
                subsets.append(p)
                
            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1
            
            self.scores_.append(scores[best])
            self.k_score_ = self.scores_[-1]
        
        return self
    def transform(self, X):
        return X[: self.indices_]
    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score
       


# In[72]:


import matplotlib.pyplot as plt


# In[73]:


from sklearn.neighbors import KNeighborsClassifier


# In[74]:


knn = KNeighborsClassifier(n_neighbors=5)


# In[75]:


sbs = SBS(knn, k_features=1)


# In[76]:


sbs.fit(X_train_std, y_train)


# In[81]:


k_feat = [len(k) for k in sbs.subsets_]


# In[89]:


plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.7, 1.02])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.tight_layout()
plt.show()


# In[90]:


knn.fit(X_train_std, y_train)


# In[91]:


print('Training accuracy:', knn.score(X_train_std, y_train))


# In[92]:


print('Test accuracy:', knn.score(X_test_std, y_test))


# In[95]:


from sklearn.ensemble import RandomForestClassifier


# In[96]:


feat_labels = df_wine.columns[1:]


# In[97]:


forest = RandomForestClassifier(n_estimators=500,
                               random_state=1)


# In[100]:


forest.fit(X_train, y_train)


# In[101]:


importances = forest.feature_importances_


# In[102]:


indices = np.argsort(importances) [::-1]


# In[105]:


for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30,
                           feat_labels[indices[f]],
                           importances[indices[f]]))


# In[115]:


plt.title('Feature Importance')
plt.bar(range(X_train.shape[1]),
        importances[indices],
        align='center')
plt.xticks(range(X_train.shape[1]),
           feat_labels[indices],
           rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout
plt.show()


# In[116]:


from sklearn.feature_selection import SelectFromModel


# In[117]:


sfm = SelectFromModel(forest, threshold=0.1, prefit=True)


# In[118]:


X_selected = sfm.transform(X_train)


# In[119]:


print('Number of features that meet this threshold',
     'criterion:', X_selected.shape[1])


# In[123]:


for f in range(X_selected.shape[1]):
            print ("%2d) %-*s %f" % (f + 1, 30,
                                      feat_labels[indices[f]],
                                    importances[indices[f]]))

