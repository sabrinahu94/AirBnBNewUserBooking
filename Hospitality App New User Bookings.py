#!/usr/bin/env python
# coding: utf-8

# ## Project 6 - Hospitality App New User Bookings

# In[1]:


## import necessary packages
import numpy as np
import pandas as pd
import time
import random
import datetime
import matplotlib.pyplot as plt
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')


# In[2]:


train_data = pd.read_csv('data/train_users_2.csv')
test_data = pd.read_csv('data/test_users.csv')
age_gender = pd.read_csv('data/age_gender_bkts.csv')
countries = pd.read_csv('data/countries.csv')


# In[3]:


print(train_data.shape)
print(train_data.columns)


# In[4]:


print(test_data.shape)
print(test_data.columns)


# In[5]:


np.setdiff1d(train_data.columns, test_data.columns)


# ## 1. Data Exploration

# In[6]:


train_data.head()


# In[7]:


test_data.head()


# In[8]:


train_data.info()


# In[9]:


test_data.info()


# In[10]:


print ('training dataset ({} rows) null value:\n'.format(train_data.shape[0]))
print (train_data.isnull().sum())
print ('\n' + '***' * 10 + '\n')
print ('test dataset ({} rows) null value:\n'.format(test_data.shape[0]))
print (test_data.isnull().sum())


# ## 2. Feature Engineering

# In[11]:


train_row = train_data.shape[0]
# The label we need to predict
labels = train_data['country_destination'].values

# the id number is not useful for prediction. We need it for submission
id_test = test_data['id']

# drop the id column
# drop the date_first_booking since there is all NaN in test dataset which we figured out in the data exploration
# drop the label in training set
train_data.drop(['country_destination', 'date_first_booking'], axis = 1, inplace = True)
test_data.drop(['date_first_booking'], axis = 1, inplace = True)


# In[12]:


full_data = pd.concat([train_data, test_data], axis = 0, ignore_index = True)


# #### date_account_created （dac）

# In[13]:


# create year, month, day feature for dac
dac = pd.to_datetime(full_data.date_account_created)
full_data['dac_year'] = np.array([x.year for x in dac])
full_data['dac_month'] = np.array([x.month for x in dac])
full_data['dac_day'] = np.array([x.day for x in dac])


# In[14]:


# create features of weekday for dac, showing Monday to Sunday
full_data['dac_wd'] = np.array([x.isoweekday() for x in dac])
df_dac_wd = pd.get_dummies(full_data.dac_wd, prefix = 'dac_wd')
full_data = pd.concat((full_data, df_dac_wd), axis = 1)
full_data.drop(['dac_wd'], axis = 1, inplace = True)


# In[15]:


def get_season(dt):
    dt = dt.date()
    # dt must be a datetime type
    if dt.month in [3,4,5]:
        return 'Spring'    
    elif dt.month in [6,7,8]:
        return 'Summer'    
    elif dt.month in [9,10,11]:
        return 'Fall'   
    else:
        return 'Winter' 


# In[16]:


# create season features from dac
full_data['dac_season'] = np.array([get_season(x) for x in dac])
df_dac_season = pd.get_dummies(full_data.dac_season, prefix = 'dac_season')
full_data = pd.concat((full_data, df_dac_season), axis = 1)
full_data.drop(['dac_season'], axis = 1, inplace = True)


# #### timestamp_first_active (tfa)

# In[17]:


tfa = full_data.timestamp_first_active.astype(str).apply(lambda x: datetime.datetime(int(x[:4]),
                                                                          int(x[4:6]), 
                                                                          int(x[6:8]),
                                                                          int(x[8:10]),
                                                                          int(x[10:12]),
                                                                          int(x[12:])))


# In[18]:


# create tfa_year, tfa_month, tfa_day feature
full_data['tfa_year'] = np.array([x.year for x in tfa])
full_data['tfa_month'] = np.array([x.month for x in tfa])
full_data['tfa_day'] = np.array([x.day for x in tfa])


# In[19]:


# create features of weekday
full_data['tfa_wd'] = np.array([x.isoweekday() for x in tfa])
df_tfa_wd = pd.get_dummies(full_data.tfa_wd, prefix = 'tfa_wd')
full_data = pd.concat((full_data, df_tfa_wd), axis = 1)
full_data.drop(['tfa_wd'], axis = 1, inplace = True)


# In[20]:


# create season features from tfa
full_data['tfa_season'] = np.array([get_season(x) for x in tfa])
df_tfa_season = pd.get_dummies(full_data.tfa_season, prefix = 'tfa_season')
full_data = pd.concat((full_data, df_tfa_season), axis = 1)
full_data.drop(['tfa_season'], axis = 1, inplace = True)


# #### Time span between dac and tfa can also be used as a feature. 

# In[21]:


dt_span = (dac - tfa).dt.seconds
full_data['dt_span'] = np.array([x for x in dt_span])


# ### Deal with Age

# In[22]:


age = full_data['age']
age.fillna(-1, inplace = True)


# In[23]:


## This are birthdays instead of age (estimating age by doing 2016 - value)
age = np.where(np.logical_and(age<2005, age>1900), 2020-age, age)
## This is the current year insted of age, we also consider this as N/A
age = np.where(np.logical_and(age<2016, age>2010), -1, age) 


# In[24]:


## Keeping ages in 14 < age < 106 as OK
interval = 15

def get_age(age):
    # age is a float number   
    if age < 0:
        return 'NA'
    elif (age < interval):
        return interval
    elif (age <= interval * 2):
        return interval*2
    elif (age <= interval * 3):
        return interval * 3
    elif (age <= interval * 4):
        return interval * 4
    elif (age <= interval * 5):
        return interval * 5
    elif (age <= interval * 6):
        return interval * 6
    elif (age <= interval * 7):
        return interval * 7
    else:
        return 'Unphysical'


# In[25]:


full_data['age'] = np.array([get_age(x) for x in age])
df_age = pd.get_dummies(full_data.age, prefix = 'age')


# In[26]:


full_data = pd.concat((full_data, df_age), axis = 1)


# ## Label Encoding and One Hot Encoding

# In[35]:


pd.set_option('display.max_columns', None) 
full_data


# In[36]:


feature_OHE = ['gender', 
               'signup_method', 
               'signup_flow', 
               'language', 
               'affiliate_channel', 
               'affiliate_provider', 
               'first_affiliate_tracked', 
               'signup_app', 
               'first_device_type', 
               'first_browser']


# **Label Encoding:**<br/>
# * Prerequisite for One-Hot-Encoding.
# * http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html<br/>

# In[37]:


from sklearn import preprocessing
LBL = preprocessing.LabelEncoder()

LE_vars=[]
LE_map=dict()
LE_map1=dict()
for cat_var in feature_OHE:
    print ("Label Encoding %s" % (cat_var))
    LE_var=cat_var+'_le'
    full_data[LE_var]=LBL.fit_transform(full_data[cat_var].fillna('none'))
    LE_vars.append(LE_var)
    LE_map1[cat_var]=dict(zip(LBL.classes_, LBL.transform(LBL.classes_))) ## Here you generate the mapping dictionary
    LE_map[cat_var]=LBL.classes_
print ("Label-encoded feaures: %s" % (LE_vars))


# In[38]:


LE_map1


# **One Hot Encoding:**<br/>
# * http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html<br/>

# In[39]:


OHE = preprocessing.OneHotEncoder(sparse=False)
start=time.time()
OHE.fit(full_data[LE_vars])
OHE_data=OHE.transform(full_data[LE_vars])
                                   
print ('One-hot-encoding finished in %f seconds' % (time.time()-start))


OHE_vars = [var[:-3] + '_' + str(level).replace(' ','_')                for var in feature_OHE for level in LE_map[var]]

print ("OHE size :" ,OHE_data.shape)
print ("One-hot encoded catgorical feature samples : %s" % (OHE_vars[:100]))


# In[40]:


full_data = pd.concat((full_data, pd.DataFrame(OHE_data,columns=OHE_vars)), axis = 1)


# In[41]:


full_data


# ## Model Building

# ### Airbnb Evaluation: NDCG

# https://www.kaggle.com/c/airbnb-recruiting-new-user-bookings#evaluation

# In[42]:


# From Kaggle Kernels

from sklearn.metrics import make_scorer
from sklearn.preprocessing import LabelBinarizer

def dcg_score(y_true, y_score, k=5):
    
    """
    y_true : array, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array, shape = [n_samples, n_classes]
        Predicted scores.
    k : int
    """
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])

    gain = 2 ** y_true - 1

    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gain / discounts)


def ndcg_score(ground_truth, predictions, k=5):
    
    """
    Parameters
    ----------
    ground_truth : array, shape = [n_samples]
        Ground truth (true labels represended as integers).
    predictions : array, shape = [n_samples, n_classes]
        Predicted probabilities.
    k : int
        Rank.
        
    Example
    -------
    >>> ground_truth = [1, 0, 2]
    >>> predictions = [[0.15, 0.55, 0.2], [0.7, 0.2, 0.1], [0.06, 0.04, 0.9]]
    >>> score = ndcg_score(ground_truth, predictions, k=2)
    1.0
    >>> predictions = [[0.9, 0.5, 0.8], [0.7, 0.2, 0.1], [0.06, 0.04, 0.9]]
    >>> score = ndcg_score(ground_truth, predictions, k=2)
    0.6666666666
    """
    lb = LabelBinarizer()
    lb.fit(range(len(predictions) + 1))
    T = lb.transform(ground_truth)

    scores = []

    # Iterate over each y_true and compute the DCG score
    for y_true, y_score in zip(T, predictions):
        actual = dcg_score(y_true, y_score, k)
        best = dcg_score(y_true, y_true, k)
        score = float(actual) / float(best)
        scores.append(score)

    return np.mean(scores)


# In[43]:


labels_le = preprocessing.LabelEncoder()
ytrain = labels_le.fit_transform(labels)
label_map = dict(zip(labels_le.classes_, labels_le.transform(labels_le.classes_)))


# In[44]:


label_map


# ### Random Forest

# Here you should use holdout validation (even though I didn't do that for simple).

# In[45]:


xtrain = full_data.drop(['id','age','date_account_created','timestamp_first_active'],axis=1).drop(feature_OHE+LE_vars,axis=1)[:train_data.shape[0]]
xtest = full_data.drop(['id','age','date_account_created','timestamp_first_active'],axis=1).drop(feature_OHE+LE_vars,axis=1)[train_data.shape[0]:]


# In[46]:


from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


# In[47]:


N_ESTIMATORS = 50
RANDOM_STATE = 2017
MAX_DEPTH = 9
RF = RandomForestClassifier(n_estimators=N_ESTIMATORS,
                                 max_depth=MAX_DEPTH,
                                 random_state=RANDOM_STATE)


# In[48]:


RF.fit(xtrain,ytrain)


# In[49]:


predict = RF.predict_proba(xtrain)


# In[50]:


np.argsort(predict[1])


# In[52]:


predict[1]


# In[54]:


k_ndcg = 5
train_ndcg_score = ndcg_score(ytrain[:1000], predict[:1000], k = k_ndcg)
train_ndcg_score


# In[64]:


id_test = test_data['id']

#Taking the 5 classes with highest probabilities
ids = []  #list of ids
cts = []  #list of countries
for i in range(len(id_test)):
    idx = id_test[i]
    ids += [idx] * 5
    cts += labels_le.inverse_transform(np.argsort(predict[i])[::-1])[:5].tolist()

#Generate submission
sub = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])
sub.to_csv('sub.csv',index=False)

