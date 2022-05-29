#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')


# In[3]:


movies.head()


# In[4]:


movies.head(1)


# In[5]:


credits.head(1)


# In[6]:


credits.head(1)


# In[7]:


movies= movies.merge(credits,on='title')


# In[8]:


movies.head(1) 


# In[9]:


#genres
#id
#keywords
#title
#overview
#cast
#crew

movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]
 



# In[10]:


movies.info()


# In[11]:


movies.head()


# In[12]:


movies.isnull().sum()


# In[13]:


movies.dropna(inplace=True)


# In[14]:


movies.duplicated().sum()


# In[15]:


movies.iloc[0].genres


# In[16]:


#'[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]'


# In[17]:


import ast


# In[18]:


def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L   


# In[19]:


movies['genres'] = movies['genres'].apply(convert)


# In[20]:


movies.head()


# In[21]:


movies['keywords'] = movies['keywords'].apply(convert)


# In[22]:


def convert3(obj):
    L = []
    counter=0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter+=1
        else:
            break
    return L  


# In[23]:


movies['cast'] = movies['cast'].apply(convert3)


# In[24]:


movies.head()


# In[25]:


movies['crew']


# In[26]:


def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job']=='Director':
            L.append(i['name'])
            break
            
    return L   


# In[27]:


movies['crew'] = movies['crew'].apply(fetch_director)


# In[28]:


movies['overview'][0]


# In[29]:


movies['overview'] = movies['overview'].apply(lambda x:x.split())


# In[30]:


movies.head()


# In[31]:


movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])


# In[32]:


movies.head()


# In[33]:


movies['tags']= movies['overview'] + movies['genres'] + movies['keywords']+movies['cast']+movies['crew']


# In[34]:


movies.head()


# In[35]:


new_df = movies[['movie_id','title','tags']]


# In[36]:


new_df['tags']=new_df['tags'].apply(lambda x:" ".join(x))


# In[37]:


new_df.head()


# In[38]:


import nltk


# In[39]:


from nltk.stem.porter import PorterStemmer
ps= PorterStemmer()


# In[40]:


def stem(text):
    y= []
    
    for i in text.split():
        y.append(ps.stem(i))
        
    return " ".join(y)    


# In[41]:


new_df['tags']=new_df['tags'].apply(stem)


# In[42]:


new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())


# In[43]:


new_df.head()


# In[44]:


new_df['tags'][0]


# In[45]:


new_df['tags'][1]


# In[46]:


get_ipython().system('pip install sklearn')


# In[47]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer (max_features=5000,stop_words='english')


# In[48]:


vectors = cv.fit_transform(new_df['tags']).toarray()


# In[49]:


vectors


# In[50]:


cv.get_feature_names()


# In[51]:


ps.stem('loving')


# In[52]:


stem('in the 22nd century, a paraplegic marine is dispatched to the moon pandora on a unique mission, but becomes torn between following orders and protecting an alien civilization. action adventure fantasy sciencefiction cultureclash future spacewar spacecolony society spacetravel futuristic romance space alien tribe alienplanet cgi marine soldier battle loveaffair antiwar powerrelations mindandsoul 3d samworthington zoesaldana sigourneyweaver jamescameron')


# In[53]:


vectors[0]


# In[54]:


get_ipython().system('pip install cosine_similarity')


# In[55]:


from sklearn.metrics.pairwise import cosine_similarity


# In[56]:


similarity = cosine_similarity(vectors)


# In[57]:


sorted(list(enumerate(similarity[0])), reverse=True, key=lambda x:x[1])[1:6]


# In[58]:


similarity[1]


# In[59]:


def recommend(movie):
    movie_index= new_df[new_df['title']==movie].index[0]
    distances = similarity[movie_index]
    movies_list= sorted(list(enumerate(distances)), reverse=True, key=lambda x:x[1])[1:6]
    
    for i in movies_list:
        print(new_df.iloc[i[0]].title)
        
    


# In[60]:


recommend ('Avatar')


# In[61]:


get_ipython().system('pip install pickle')


# In[62]:


import pickle


# In[63]:


pickle.dump(new_df,open('movies.pkl','wb'))


# In[64]:


pickle.dump(new_df.to_dict(), open('movies_dict.pkl','wb'))


# In[65]:


pickle.dump(similarity,open('similarity.pkl','wb'))


# In[ ]:




