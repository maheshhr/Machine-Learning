#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[82]:


messages = pd.read_csv('SMSSpamCollection', sep='\t', names=['label','message'])


# In[83]:


messages.head()


# In[84]:


messages.shape


# In[85]:


count = 0
for i in messages['label']:
    if i == 'spam':
        count+=1
print(count)


# In[86]:


import string
from nltk.corpus import stopwords


# In[87]:


def text_process(text):
    nopunc = [i for i in text if i not in string.punctuation ]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


# In[88]:


import nltk
nltk.download('stopwords')


# In[89]:


text_process("Hello my Dear...!!! how are you???")


# In[90]:


messages['message'] = messages['message'].apply(text_process)


# In[92]:


messages.head()


# In[93]:


from nltk.stem  import WordNetLemmatizer


# In[94]:


wnl = WordNetLemmatizer()


# In[95]:


import nltk
nltk.download('wordnet')


# In[96]:


def process2(tex):
    relist = []
    for i in tex:
        lower = i.lower()
        relist.append(wnl.lemmatize(lower))
    return relist
        


# In[97]:


messages['message'] = messages['message'].apply(process2)


# In[98]:


messages.head()


# In[99]:


in_data = messages.loc[ : 5000]


# In[100]:


out_data = messages.loc[5000:]


# In[101]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[102]:


def reconstruct(x):
    resstr = ' '.join(x)
    return resstr


# In[103]:


in_data['message'] = in_data['message'].apply(reconstruct)


# In[104]:


in_data.head()


# In[105]:


tf = TfidfVectorizer(lowercase=False)


# In[106]:


tf.fit(in_data['message'])


# In[107]:


len(tf.vocabulary_)


# In[108]:


tf.vocabulary_


# In[109]:


data = tf.transform(in_data['message']).toarray()


# In[111]:


data.shape


# In[112]:


from sklearn.ensemble import RandomForestClassifier


# In[113]:


rfc = RandomForestClassifier()


# In[114]:


rfc.fit(data,in_data['label'])


# In[115]:


out_data['message'] = out_data['message'].apply(reconstruct)


# In[116]:


out_data['message'].head()


# In[117]:


test_input = tf.transform(out_data['message']).toarray()


# In[118]:


test_input


# In[119]:


test_input.shape


# In[120]:


prediction = rfc.predict(test_input)


# In[121]:


prediction


# In[122]:


from sklearn.metrics import accuracy_score


# In[123]:


accuracy_score(prediction,out_data['label'])


# In[124]:


from sklearn.metrics import  classification_report


# In[125]:


print(classification_report(prediction,out_data['label']))


# In[ ]:




