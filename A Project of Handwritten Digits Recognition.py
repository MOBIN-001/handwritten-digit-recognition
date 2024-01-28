#!/usr/bin/env python
# coding: utf-8

# ## IMPORT REQ LIBRARIES

# In[1]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.datasets import load_digits 
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.datasets import fetch_openml 
from sklearn.metrics import accuracy_score, classification_report 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn import svm
import tensorflow as tf


# ## Now load datasets.

# In[2]:


mnist= fetch_openml('mnist_784',parser='auto')


# In[3]:


print(mnist.DESCR)


# In[4]:


mnist.data


# In[5]:


mnist.target.shape


# In[6]:


mnist.target


# In[7]:


mnist


# In[8]:


x,y= mnist['data'],mnist['target']


# In[9]:


x.shape, x.dtypes


# In[10]:


y.shape


# In[11]:


some_digit=x.to_numpy()[36003]
some_digit_image= some_digit.reshape(28,28)
plt.imshow(some_digit_image, cmap="binary")
plt.axis("off")
plt.show()


# In[12]:


y[36003]


# In[13]:


x_train,x_test= x[:60000],x[60000:]


# In[14]:


y_train,y_test= y[:60000],y[60000:]


# In[15]:


shuffle_index=np.random.permutation(60000)
x_train, y_train= x_train.loc[shuffle_index],y_train.loc[shuffle_index]


# ## Implementing SVM algorithm and calculating its accuracy.

# In[16]:


svc=svm.SVC(gamma="scale", class_weight="balanced",C=100)
svc.fit(x_train, y_train)
result=svc.predict(x_test)
print('accuracy:', accuracy_score(y_test,result))
print(classification_report(y_test,result))


# In[ ]:




