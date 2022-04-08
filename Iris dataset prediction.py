#!/usr/bin/env python
# coding: utf-8

# In[23]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[24]:


df = pd.read_csv(r"C:\Users\Arush\Downloads\IRIS.csv")


# In[25]:


df.head()


# In[26]:


df


# In[27]:


#df['species'] = df['species'].str.split (' ').str.join


# In[28]:


#df['species'].str.slice(5)


# In[29]:


df['species'] = df['species'].str.slice(5)


# In[30]:


df.head()


# In[31]:


df['species']


# In[32]:


df.head()


# In[33]:


df


# In[34]:


df.info()


# In[35]:


df.head()


# In[36]:


y = df[['sepal_length']]


# In[37]:


x = df[['sepal_width']]


# In[38]:


from sklearn.model_selection import train_test_split


# In[39]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3)


# In[40]:


x_train.head()


# In[41]:


x_test.head()


# In[42]:


y_train.head()


# In[43]:


y_train.head()


# In[44]:


from sklearn.linear_model import LinearRegression


# In[45]:


lr = LinearRegression()


# In[46]:


lr.fit(x_train,y_train)


# In[47]:


y_pred = lr.predict(x_test)


# In[48]:


y_test.head()


# In[49]:


y_pred[0:5]


# In[50]:


from sklearn.metrics import mean_squared_error


# In[51]:


mean_squared_error(y_test,y_pred)


# # multiple Linear Regression
# 

# In[52]:


y = df[['sepal_length']]


# In[53]:


x = df[['sepal_width','petal_length','petal_width']]


# In[54]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3)


# In[55]:


lr2 = LinearRegression()


# In[56]:


lr2.fit(x_train,y_train)


# In[57]:


y_pred = lr2.predict(x_test)


# In[58]:


mean_squared_error(y_test,y_pred)


# In[59]:


print ("Aroosh Thapliyal")


# In[64]:


print(np.pi)


# In[65]:


np.pi


# In[66]:


2 * np.pi


# In[67]:


x = np.arange(0,2*np.pi,0.1)
y = np.sin(x)


# In[68]:


plt.plot(x,y)
plt.show()


# In[73]:


x = np.arange(0,4*np.pi,0.1)
y = np.sin(x)
plt.plot(x,y)
plt.show()


# In[79]:


import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-np.pi,np.pi,100)

p = 4*np.sin(x) 
q = 2*np.sin(x) 
r = np.sin(x)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

plt.plot(x,p, 'b-' , label='y=2sin(x)')
plt.plot(x,q, 'c-', label='y=sin(x)')
plt.plot(x,r, 'm-', label='y=sin(x/2)')

plt.legend(loc='upper left')

plt.show()


# In[ ]:




