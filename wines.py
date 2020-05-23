#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv('wines.csv')


# In[3]:


df        # here wine has three classes which depends on its contents. we need to predict the class of wine


# In[4]:


df.head()


# In[5]:


y = df['Class']


# In[6]:


y.value_counts()      # no. of 2s , 1s and 3s in y


# In[7]:


y_cat = pd.get_dummies(y)


# In[8]:


y_cat                    # as this is the output we need all the dummy columns so we don't remove any dummy variable here


# In[9]:


X = df.drop('Class', axis =1 )          # removed class column from X as class is output not a feature


# In[10]:


X


# In[11]:


X.info()


# In[12]:



# In[14]:


from keras.models import Sequential


# In[15]:


model = Sequential()


# In[16]:


from keras.layers import Dense        # we have 13 features here


# In[17]:


model.add(Dense(units=5,input_shape=(13,),
                activation='relu',
                kernel_initializer="he_normal"))  # he_normal is initializer used
# input_shape is taken (13,) since columns are 13


# In[18]:


model.summary()


# In[19]:


model.add(Dense(units=8,input_shape=(13,),
                activation='relu',
                kernel_initializer="he_normal"))


# In[20]:


model.add(Dense(units=2,input_shape=(13,),                  
                activation='relu',
                kernel_initializer="he_normal"))     # In multiclassification output neurons should be taken = no. of categories of y


# In[21]:


model.summary()


# In[22]:


model.add(Dense(units=3, activation='softmax'))


# In[23]:


model.summary()


# In[24]:


from keras.optimizers import RMSprop


# In[25]:


model.compile(optimizer= RMSprop(learning_rate=0.001),loss='categorical_crossentropy',metrics=['accuracy'])

# plugging everything together, optimizer to change weights acc. to loss, using RMSprop optimizer, in multi-classification ,
# error function used is categorical_crossentropy  since the output y is categorical.


# In[26]:


model.layers


# In[27]:


model.layers[0]


# In[28]:


model.layers[0].input  # shows 13 inputs in first layer


# In[29]:


model.layers[3].output   # in last layer 2 outputs


# In[39]:


a= model.fit(X,y_cat,epochs = 100)   # weights change again after every epoch , in each epoch the weights are calculated for each feature


# In[31]:


import keras.backend as k


# In[32]:


k.clear_session()           #reinitializes the weights, sometimes helpful


# In[33]:


model.get_weights()          # gives weights calculated in each epoch


# In[34]:


model.save('wineclasspred.h5')






# In[41]:


a.history













