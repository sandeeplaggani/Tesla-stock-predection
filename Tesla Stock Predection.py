#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv("TSLA.csv")


# In[3]:


df_Train = df[0:1600]
df_test = df[1600:]


# In[4]:


df_Train


# In[5]:


df_test


# In[6]:


train = df_Train.iloc[:, 4:5].values
test_actual = df_test.iloc[:, 4:5].values


# In[7]:


##Considering closing value of stocks so extracting the close column
train


# In[8]:


test_actual


# In[9]:


######NORmalizing the train values

from sklearn.preprocessing import MinMaxScaler
mm_sc = MinMaxScaler(feature_range = (0, 1))
training = mm_sc.fit_transform(train)


# In[10]:


training


# In[11]:


X_train = []
y_train = []
for i in range(60, 1600):
    X_train.append(training[i-60:i, 0])
    y_train.append(training[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)


# In[12]:


X_train.shape


# In[13]:


X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


# In[14]:


X_train


# In[15]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout


# In[16]:


Stock_pred = Sequential()

Stock_pred.add(LSTM(units = 60, return_sequences = True, input_shape = (X_train.shape[1], 1)))
Stock_pred.add(Dropout(0.1))

Stock_pred.add(LSTM(units = 60, return_sequences = True))
Stock_pred.add(Dropout(0.1))

Stock_pred.add(LSTM(units = 60, return_sequences = True))
Stock_pred.add(Dropout(0.1))

Stock_pred.add(LSTM(units = 60))
Stock_pred.add(Dropout(0.1))

Stock_pred.add(Dense(units = 1))

Stock_pred.compile(optimizer = 'adam', loss = 'mean_squared_error')
##Training the model
Stock_pred.fit(X_train, y_train, epochs = 25, batch_size = 16)


# In[17]:


test_head = train[-60:]


# In[18]:


test_head


# In[19]:


test_actual.shape


# In[20]:


##For getting the 1st predicted value, we have o concatenate the last 60 time steps from the training dataset
test_head.shape


# In[21]:


Test = np.concatenate((test_head,test_actual), axis=0)


# In[22]:


Test.shape


# In[23]:


Test


# In[24]:


Test = mm_sc.transform(Test)
X_test = []
for i in range(60, 152):
    X_test.append(Test[i-60:i, 0])
X_test = np.array(X_test)


# In[25]:


X_test


# In[26]:


X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


# In[27]:


X_test


# In[28]:


y_pred = Stock_pred.predict(X_test)
y_pred = mm_sc.inverse_transform(y_pred)


# In[29]:


plt.plot(test_actual, color = 'Green', label = 'tesla Actual stock closing price')
plt.plot(y_pred, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Tesla Stock price Prediction')
plt.xlabel('Date')
plt.ylabel('Tesla Closing stock')
plt.legend()
plt.show()


# In[ ]:




