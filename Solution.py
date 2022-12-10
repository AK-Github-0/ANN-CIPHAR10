#!/usr/bin/env python
# coding: utf-8

# Importing Libraries

# In[104]:


import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation,Dropout
from tensorflow.keras.utils import to_categorical,plot_model
from tensorflow.keras.datasets import cifar10
from matplotlib import pyplot as pl


# Using unpickle function to make train and test data

# In[147]:


import numpy as np
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
X = unpickle("data_batch_1") 
X_train = X[b'data']
X_test = np.array(X[b'labels']).reshape((-1,1))

X_train.shape,Y_train.shape


# In[148]:


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
Z = unpickle("test_batch") 
Y_train = Z[b'data']
Y_test = np.array(Z[b'labels']).reshape((-1,1))

Y_test.shape, Y_test.shape


# In[149]:


input_size = 32*32


# In[152]:


X_train = np.reshape(X_train,[-1,input_size])
X_train = X_train.astype('float32') 
X_test = X_test.astype('float32') 
X_train / 255 
X_test / 255 
Y_train = np_utils.to_categorical(y_train, 10) 
Y_test = np_utils.to_categorical(y_test, 10)


# In[166]:


model = Sequential()
model.add(Dense(150,input_dim = 32*32))
model.add(Dropout(0.1))
model.add(Dense(150, activation='tanh'))
model.add(Dropout(0.1))
model.add(Dense(10, activation='softmax'))


# In[167]:


model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])


# In[168]:


model.summary()


# In[169]:


history = model.fit(X_train, Y_train, batch_size=32, epochs=10, validation_split = 0.2)


# In[170]:


plt.plot(history.history['accuracy'])
plt.show()


# In[172]:





# In[ ]:




