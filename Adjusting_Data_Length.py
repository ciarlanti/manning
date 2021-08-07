#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"></ul></div>

# In[1]:


from __future__ import absolute_import, division, print_function
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# In[2]:


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(
    path='imdb.npz',
    num_words=None,
    skip_top=0,
    maxlen=None,
    seed=113,
    start_char=1,
    oov_char=2,
    index_from=3
)


# In[3]:


unique, counts = np.unique(y_train, return_counts=True)
positive_sample_count = counts[1]


# Following is all the  already done part from milestone 2

# In[4]:


idx = np.argwhere(y_train>0) # Select positive comment's index in training data
np.random.seed(seed=100) # use seed to ensure selected records are always same
np.random.shuffle(idx) #Shuffle it at random
FRAC = 0.25
idxs = idx[:int(len(idx)*FRAC)]
y_trains = y_train[idxs]
x_trains = x_train[idxs]

# preserve negative cases
idxn = np.argwhere(y_train==0)
x_train0 = x_train[idxn]
y_train0 = y_train[idxn]

temp_idx = np.arange(0, len(idxs), 1)
temp_idx_multi = np.tile(temp_idx, (1, 10)).T # shape is (31250, 1)
np.random.seed(seed=200) # rerun from temp_idx_multi definition cell to ensure same result
np.random.shuffle(temp_idx_multi)
sample_idx = temp_idx_multi[:positive_sample_count].squeeze(1)
x_train[sample_idx] # no point of using just the positive, since we are going by indeces anyway


# And now we process the data to have all the items (reviews) having the same number of tokens (words).

# In[5]:


x_trains_os = x_train[sample_idx] # again, we are using indeces
y_trains_os = y_train[sample_idx]
x_train_assembled = np.concatenate((x_train0, x_trains_os), axis = None)
y_train_assembled = np.concatenate((y_train0, y_trains_os), axis = None)
shuffled_idx = np.arange(0, len(y_train_assembled), 1)
np.random.seed(seed=300)
np.random.shuffle(shuffled_idx)
print(shuffled_idx.shape)


# In[6]:


shuffled_idx


# In[7]:


x_train_assembled_shuffled = x_train_assembled[shuffled_idx]
y_train_assembled_shuffled = y_train_assembled[shuffled_idx]
print(x_train_assembled_shuffled.shape, y_train_assembled_shuffled.shape)


# In[8]:


word_index = tf.keras.datasets.imdb.get_word_index()
word_index = {k:(v+3) for k,v in word_index.items()} 
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

index_word = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(encoded_array):
    return ' '.join([index_word.get(i, '?') for i in encoded_array])


# In[9]:


train_data = tf.keras.preprocessing.sequence.pad_sequences(x_train_assembled_shuffled,
                                                        value=word_index["<PAD>"],
                                                        padding='pre',
                                                        maxlen=256)

test_data = tf.keras.preprocessing.sequence.pad_sequences(x_test,
                                                       value=word_index["<PAD>"],
                                                       padding='pre',
                                                       maxlen=256)


# In[10]:


train_data.shape


# In[11]:


test_data.shape

