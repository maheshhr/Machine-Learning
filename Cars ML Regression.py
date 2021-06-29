#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[4]:


auto = pd.read_csv('auto.txt')


# In[5]:


auto.head()


# In[6]:


import numpy as np


# In[7]:


auto = auto.replace('?',np.nan)


# In[8]:


auto.head()


# In[9]:


auto.describe(include='all')


# In[10]:


auto['price'].describe()


# In[11]:


auto['price'] = pd.to_numeric(auto['price'], errors='coerce')


# In[12]:


auto['price'].describe()


# In[15]:


auto = auto.drop('normalized-losses',axis=1)


# In[16]:


auto.head()


# In[17]:


auto.describe(include='all')


# In[18]:


auto['horsepower'].describe()


# In[21]:


auto['horsepower'] = pd.to_numeric(auto['horsepower'],errors='coerce')


# In[22]:


auto['horsepower'].describe()


# In[23]:


auto['num-of-cylinders'].describe()


# In[25]:


dict1={
    'two':2,
    'three':3,
    'four':4,
    'five':5,
    'six':6,
    'eight':8,
    'twelve':12
    
}


# In[27]:


auto['num-of-cylinders'].replace(dict1,inplace=True)


# In[28]:


auto.head()


# In[29]:


auto = pd.get_dummies(auto, columns=['make','fuel-type','aspiration','num-of-doors','body-style','drive-wheels',
    'engine-location','engine-type','fuel-system'])


# In[30]:


auto.head()


# In[31]:


auto = auto.dropna()


# In[33]:


auto


# In[34]:


auto[auto.isnull().any(axis=1)]


# In[36]:


auto.columns


# In[37]:


from sklearn.model_selection import train_test_split


# In[38]:


data_input = auto.drop('price',axis=1)


# In[39]:


data_output = auto['price']


# In[41]:


data_input_train, data_input_test, data_output_train, data_output_test = train_test_split(data_input,data_output, test_size=0.2,random_state=0)


# In[42]:


from sklearn.linear_model import LinearRegression


# In[43]:


lr = LinearRegression()


# In[44]:


lr.fit(data_input_train, data_output_train)


# In[46]:


lr.score(data_input_train,data_output_train)


# In[49]:


lr.coef_


# In[50]:


predicts = data_input_train.columns
coef = pd.Series(lr.coef_,predicts).sort_values()


# In[51]:


coef #-ve values means this feture car is cheaper


# In[52]:


coef[0:30]


# In[53]:


coef[31:]


# In[54]:


result = lr.predict(data_input_test)


# In[57]:


get_ipython().run_line_magic('pylab', 'inline')
pylab.rcParams['figure.figsize'] = (15,6)

plt.plot(result, label='Predicted')
plt.plot(data_output_test.values, label='Actual')

plt.ylabel('price')
plt.legend()
plt.show()


# In[58]:


r_square = lr.score(data_input_test,data_output_test)


# In[59]:


r_square


# In[61]:


from sklearn.metrics import mean_squared_error


# In[62]:


mse = mean_squared_error(result, data_output_test)


# In[63]:


mse


# In[64]:


import math


# In[65]:


math.sqrt(mse) # 5108 amount deiffernce will be there whether may be high or low


# In[66]:


from sklearn.linear_model import Lasso


# In[114]:


ls =Lasso(alpha=0.55, normalize=True)


# In[115]:


ls.fit(data_input_train,data_output_train)


# In[116]:


ls.score(data_input_train, data_output_train)


# In[117]:


coef = pd.Series(ls.coef_ , predicts).sort_values()


# In[118]:


coef


# In[119]:


coef[0:30]


# In[120]:


coef[31:]


# In[121]:


result = ls.predict(data_input_test)


# In[122]:


result


# In[123]:


pylab.rcParams['figure.figsize'] = (15,6)

plt.plot(result, label='Predicted')
plt.plot(data_output_test.values, label='Actual')
plt.ylabel('Price')

plt.legend()
plt.show()


# In[124]:


r_square = ls.score(data_input_test, data_output_test)


# In[125]:


r_square


# In[126]:


mse = mean_squared_error(result, data_output_test)


# In[127]:


mse


# In[128]:


math.sqrt(mse)


# In[129]:


from sklearn.linear_model import Ridge


# In[160]:


rd = Ridge(alpha=0.5,normalize=True)


# In[161]:


rd.fit(data_input_train, data_output_train)


# In[162]:


rd.score(data_input_train, data_output_train)


# In[163]:


coef = pd.Series(rd.coef_, predicts).sort_values()


# In[164]:


coef


# In[165]:


coef[:30]


# In[166]:


coef[31:]


# In[167]:


result = rd.predict(data_input_test)


# In[168]:


result


# In[169]:


pylab.rcParams['figure.figsize'] = (15,6)

plt.plot(result, label='Predicted')
plt.plot(data_output_test.values, label='Actual')
plt.ylabel('Price')

plt.legend()
plt.show()


# In[170]:


r_square = rd.score(data_input_test,data_output_test)


# In[171]:


r_square


# In[172]:


mse = mean_squared_error(result, data_output_test)


# In[173]:


mse


# In[174]:


math.sqrt(mse)


# ## finally ridge model is selected

# In[ ]:




