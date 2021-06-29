#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[7]:


df = pd.read_excel('\\Data Science\\dataset pr\\pima-data.xlsx') #reading data set from my local directory


# In[9]:


df


# In[11]:


df.shape


# In[12]:


df.columns


# In[13]:


df = df.drop('has_diabetes',axis=1) #droping unwanted column


# In[14]:


df.head()


# In[16]:


df.dtypes


# In[17]:


output_dict = {True:1, False:0} 


# In[18]:


df['diabetes'] = df['diabetes'].map(output_dict) #replacing text with number


# In[20]:


df.head()


# In[21]:


df.isna().sum() #checking for null values


# In[22]:


def fillinsulin(num):
    if num == 0:
        return df['insulin'].mean()
    else:
        return num


# In[24]:


df['insulin'] = df['insulin'].apply(fillinsulin) #generally insulin must not be 0 so applying mean value


# In[25]:


df.head()


# In[26]:


df.corr() #finding corellation b/w columns which will be an effect on result


# In[27]:


import seaborn as sb


# In[28]:


sb.heatmap(df.corr(),annot=True)


# In[30]:


from matplotlib import pyplot as plt


# In[31]:


plt.figure(figsize = (15,10))
sb.heatmap(df.corr(),annot=True)
plt.show()


# In[32]:


df.head()


# In[33]:


df = df.drop(columns = ['thickness','diabetes_orig'],axis=1) #droping coreelation columns


# In[34]:


df.head()


# In[35]:


plt.figure(figsize = (15,10))
sb.heatmap(df.corr(),annot=True)
plt.show()


# In[36]:


from sklearn.model_selection import train_test_split


# In[37]:


input_col = df.drop('diabetes',axis=1) #spilting as an input and output column


# In[38]:


input_col


# In[39]:


output_col = df['diabetes']


# In[40]:


output_col


# In[41]:


input_train, input_test, output_train, output_test = train_test_split(input_col,output_col,test_size=0.2)


# In[43]:


input_train.shape #splitting data for trainig 80% and testing 20%  


# In[44]:


output_train.shape


# In[45]:


input_test.shape


# In[46]:


output_test.shape


# In[48]:


from sklearn.linear_model import LogisticRegression


# In[49]:


lr = LogisticRegression()


# In[50]:


lr.fit(input_train,output_train) #training the model


# In[51]:


predicts = lr.predict(input_test) #testing the model


# In[52]:


predicts


# In[54]:


from sklearn.metrics import accuracy_score


# In[55]:


accuracy_score(predicts,output_test) #checking accuracy


# In[56]:


0.8181*154


# In[58]:


from sklearn.naive_bayes import GaussianNB


# In[59]:


gb = GaussianNB()


# In[60]:


gb.fit(input_train,output_train)


# In[69]:


predicts = gb.predict(input_test)


# In[70]:


predicts


# In[71]:


accuracy_score(predicts,output_test)


# In[72]:


0.7987*154


# In[75]:


from sklearn.neighbors import KNeighborsClassifier


# In[77]:


kn = KNeighborsClassifier()


# In[78]:


kn.fit(input_train,output_train)


# In[80]:


prediction = kn.predict(input_test)


# In[81]:


prediction


# In[82]:


accuracy_score(prediction,output_test)


# In[83]:


0.7467*154


# In[84]:


from sklearn.tree import DecisionTreeClassifier


# In[85]:


dt = DecisionTreeClassifier()


# In[86]:


dt.fit(input_train,output_train)


# In[87]:


result = dt.predict(input_test)


# In[88]:


result


# In[89]:


accuracy_score(result,output_test)


# In[90]:


0.7468*154


# In[92]:


from sklearn.svm import SVC


# In[93]:


sv = SVC()


# In[94]:


sv.fit(input_train,output_train)


# In[95]:


predicts = sv.predict(input_test)


# In[96]:


predicts


# In[97]:


accuracy_score(predicts,output_test)


# In[98]:


0.7987*154


# In[99]:


from sklearn.ensemble import RandomForestClassifier


# In[100]:


rf = RandomForestClassifier()


# In[101]:


rf.fit(input_train,output_train)


# In[102]:


predicts = rf.predict(input_test)


# In[103]:


predicts


# In[104]:


accuracy_score(predicts,output_test)


# In[105]:


0.7857*154


# ## so applying some of the best ML algorithm i got max accuracy score in logistic regression, svc and GaussianNB so for the final production i am suggesting LOGISTIC REGRESSION MODEL
