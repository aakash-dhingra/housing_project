#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


housing = pd.read_csv('housingnew.csv')


# In[3]:


housing.head(5)


# In[4]:


housing.info()


# In[5]:


housing['CHAS'].value_counts()


# In[6]:



    #import numpy as np
    #def train_test_split(data,test_ratio):
    #shuffled=np.random.permutation(len(data))
    #print(shuffled)
    #test_set_size=int(len(data))*test_ratio
    #test_indicies=shuffled[:test_set_size]
    #train_indicies=shuffled[test_set_size:]
    #return data.iloc[train_indicies],data.iloc[test_indicies]


# In[7]:


import numpy as np
#train_set , test_set = train_test_split(housing,0.2)


# In[8]:


from sklearn.model_selection import train_test_split
train_set,test_set=train_test_split(housing,test_size=0.2,random_state=42)
   #print(f"rows in train set: {len(train_set)}\n rows in test set: {len(test_set)} ")


# In[9]:


print(len(train_set))


# In[10]:


print(len(test_set))


# In[11]:


from sklearn.model_selection import StratifiedShuffleSplit
split=StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index,test_index in split.split(housing,housing['CHAS']):
        strat_train_set=housing.loc[train_index]
        strat_test_set=housing.loc[test_index]


# In[12]:


strat_test_set['CHAS'].value_counts()


# In[13]:


housing=strat_train_set.copy()


# ##next haeding is

# ## corelate matrix
# 
# 
# 

# In[14]:


import matplotlib.pyplot as plt
corr_matrix=housing.corr()


# In[15]:


corr_matrix['MEDV'].sort_values(ascending=False)


# In[16]:


#housing= strat_train_set.drop("MEDV",axis=1)
#housing_labels= strat_train_set["MEDV"].copy()


# In[17]:


from pandas.plotting import scatter_matrix
attributes=['MEDV','RM','ZN','LSTAT']
scatter_matrix(housing[attributes],figsize=(12,8))


# In[18]:


housing.plot(kind="scatter",x="RM",y="MEDV",alpha=0.8)


# In[19]:


#housing= strat_train_set.drop("MEDV",axis=1)
#housing_labels= strat_train_set["MEDV"].copy()


# ## attribute combination

# In[20]:


housing['TAXRM']=housing['TAX']/housing['RM']


# In[21]:


#median=housing["RM"].median()


# In[22]:


#housing= strat_train_set.drop("MEDV",axis=1)
#housing_labels= strat_train_set["MEDV"].copy()


# In[23]:


#from sklearn.impute import SimpleImputer
#imputer=SimpleImputer(strategy="median")
#imputer.fit(housing)
#X=imputer.transform(housing)


# In[24]:


#X=imputer.transform(housing)


# In[25]:


#housing_tr=pd.DataFrame(X , columns=housing.columns)


# In[26]:


#housing_tr.describe()


# In[27]:


housing.head(5)


# In[28]:


corr_matrix=housing.corr()
corr_matrix["MEDV"].sort_values(ascending=False)


# In[29]:


#housing= strat_train_set.drop("MEDV",axis=1)
 #housing_labels= strat_train_set["MEDV"].copy()


# In[30]:


housing.plot(kind="scatter",x="TAXRM",y="MEDV",alpha=0.8)


# In[31]:


housing= strat_train_set.drop("MEDV",axis=1)
housing_labels= strat_train_set["MEDV"].copy()


# In[32]:


from sklearn.impute import SimpleImputer
imputer=SimpleImputer(strategy="median")
imputer.fit(housing)


# In[33]:


X=imputer.transform(housing)


# In[34]:


housing_tr=pd.DataFrame(X , columns=housing.columns)


# In[35]:


housing_tr.describe()



# In[36]:


median=housing["RM"].median()


# In[37]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline=Pipeline([
    ('imputer',SimpleImputer(strategy='median')),
    ('std_scalar',StandardScaler()),
])


# In[38]:


housing_num_tr=my_pipeline.fit_transform(housing_tr)


# In[39]:


print(housing_num_tr)


# #select a desired model

# In[40]:


housing_num_tr=my_pipeline.fit_transform(housing)


# In[41]:


housing_num_tr.shape


# # select a desired model

# In[42]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
model= RandomForestRegressor()
#model=LinearRegression()
#model=DecisionTreeRegressor()
model.fit(housing_num_tr,housing_labels)


# In[43]:


some_data=housing.iloc[:5]


# In[44]:


some_labels=housing_labels.iloc[:5]


# In[45]:


prepared_data=my_pipeline.transform(some_data)


# In[46]:


model.predict(prepared_data)


# In[47]:


some_labels


# In[48]:


list(some_labels)


# # Evaluating model

# In[49]:


from sklearn.metrics import mean_squared_error
housing_predictions=model.predict(housing_num_tr)
lin_mse=mean_squared_error(housing_labels,housing_predictions)
lin_rmse=np.sqrt(lin_mse)


# In[50]:


lin_mse


# # using better evaluation by cross validation

# In[51]:


from sklearn.model_selection import cross_val_score
scores=cross_val_score(model,housing_num_tr,housing_labels,scoring="neg_mean_squared_error",cv=10)
rmse_scores=np.sqrt(-scores)


# In[52]:


rmse_scores


# In[53]:


def print_scores(scores):
    print("score:",scores)
    print("mean:",scores.mean())
    print("standard deviation",scores.std())


# In[54]:


print_scores(rmse_scores)


# In[ ]:




