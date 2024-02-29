#!/usr/bin/env python
# coding: utf-8

# In[34]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import style
style.use('ggplot')

from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk
stop_words = set(stopwords.words('english'))
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix, ConfusionMatrixDisplay


# In[87]:


get_ipython().system('pip install processing')


# In[36]:


nltk.download('stopwords')


# In[37]:


data=pd.read_csv("C:/Users/HP/OneDrive/Documents/archive_(4)[1]/vaccination_tweets.csv")


# In[38]:


data


# In[86]:


data.info


# In[40]:


data.isnull().sum()


# In[41]:


data.columns


# In[42]:


text_data = data.drop(['id', 'user_name', 'user_location', 'user_description', 'user_created',
       'user_followers', 'user_friends', 'user_favourites', 'user_verified',
       'date','hashtags', 'source', 'retweets', 'favorites',
       'is_retweet'],axis=1)
text_data.head()


# In[43]:


print(text_data['text'].iloc[0],"/n")
print(text_data['text'].iloc[1],"/n")
print(text_data['text'].iloc[2],"/n")
print(text_data['text'].iloc[3],"/n")
print(text_data['text'].iloc[4],"/n")


# In[44]:


text_data.info()


# In[49]:


def data_processing(text):
    text=text.lower()
    text=re.sub(r"https\S+|www\S+",'',text,flags=re.MULTILINE)
    text=re.sub(r'\@w+|\#','',text)
    text=re.sub(r'[^\w\s]','',text)
    text_tokens=word_tokenize(text)
    filtered_text=[w for w in text_tokens if not w in stop_words]
    return " ",join(filtered_text)
    


# In[ ]:





# In[66]:


text_data = text_data.drop_duplicates('text')


# In[67]:


stemmer = PorterStemmer()
def stemming(data):
   text=[stemmer.stem(word)for word in data]
   return data


# In[68]:


text_data['text'] = text_data['text'].apply(lambda x: stemming(x))


# In[69]:


text_data.head()


# In[70]:


print(text_data['text'].iloc[0],"/n")
print(text_data['text'].iloc[1],"/n")
print(text_data['text'].iloc[2],"/n")
print(text_data['text'].iloc[3],"/n")
print(text_data['text'].iloc[4],"/n")


# In[71]:


text_data.info


# In[72]:


def polarity(text):
    return TextBlob(text).sentiment.polarity


# In[73]:


text_data['polarity']=text_data['text'].apply(polarity)


# In[74]:


text_data.head(10)


# In[75]:


def sentiment(label):
    if label <0:
        return "Negative"
    elif label ==0:
        return "Neutral"
    elif label>0:
        return "Positive"


# In[76]:


text_data['sentiment'] = text_data['polarity'].apply(sentiment)


# In[78]:


text_data.head()


# In[79]:


fig = plt.figure(figsize=(5,5))
sns.countplot(x='sentiment',data=text_data)


# In[88]:


pos_tweets = text_data[text_data.sentiment == 'Positive']
pos_tweets = pos_tweets.sort_values(['polarity'],ascending=False)
pos_tweets.head()


# In[89]:


pos_tweets = text_data[text_data.sentiment == 'Negative']
pos_tweets = pos_tweets.sort_values(['polarity'],ascending=True)
pos_tweets.head()


# In[ ]:




