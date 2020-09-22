#!/usr/bin/env python
# coding: utf-8




from __future__ import absolute_import, division, print_function

# Import TensorFlow >= 1.10 and enable eager execution
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
tf.enable_eager_execution()



from sklearn.model_selection import train_test_split

import unicodedata
import re
import numpy as np
import os
import time
import config 

os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda_device
#print(tf.__version__)
from tensorflow.python.client import device_lib
#print (device_lib.list_local_devices())
import random

from preprocess import load_dataset
from model import Encoder,style_model,Decoder,save_list,load_list

import linecache
import pickle

import warnings
warnings.filterwarnings("ignore")




# Download the file
dataset_name=config.dataset
mode=config.mode
EPOCHS = config.epoch
path_to_file ='data/'+dataset_name+"/question_answer_user.txt"


# In[4]:


def save_list(list_name,path_str):
    list_file=open(str(path_str),'wb')
    pickle.dump(list_name,list_file)
    list_file.close()
def load_list(path_str):
    list_file=open(str(path_str),'rb')
    list_name=pickle.load(list_file)
    return list_name

num_examples = 10000
target_tensor, news_tensor, targ_lang,news_lang, max_length_targ,max_length_news, = load_dataset(path_to_file, num_examples)


vocab_news_size = len(news_lang.word2idx)
vocab_tar_size = len(targ_lang.word2idx)

# In[14]:

#读取
list_file = open('w_d/'+dataset_name+'/w_d_list.pickle','rb')
w_d_list = pickle.load(list_file)


# In[10]:


# Creating training and validation sets using an 80-20 split
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val,news_tensor_train,news_tensor_val = train_test_split(w_d_list, target_tensor,news_tensor, test_size=0.2,random_state=1)

# Show length
len(input_tensor_train), len(target_tensor_train),len(news_tensor_train), len(input_tensor_val), len(target_tensor_val),  len(news_tensor_val)


# In[17]:


BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE =config.batch_size
N_BATCH = BUFFER_SIZE//BATCH_SIZE
embedding_dim = 256
topic_num=64
units = 512

dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train,news_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)



encoder1 = Encoder(vocab_news_size, embedding_dim, units, BATCH_SIZE)
encoder2=style_model( topic_num)
decoder = Decoder(vocab_tar_size, embedding_dim, units+topic_num, BATCH_SIZE)


# ## Define the optimizer and the loss function




optimizer = tf.train.AdamOptimizer(learning_rate=0.001)


def loss_function(real, pred):
  mask = 1 - np.equal(real, 0)
  loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred) * mask
  return tf.reduce_mean(loss_)


# ## Checkpoints (Object-based saving)

# In[67]:

if mode=='forward':
    print("我在这儿")
    checkpoint_dir = './checkpoints/'+dataset_name 
elif mode=='backward':
    checkpoint_dir = './r_checkpoints/'+dataset_name 
else:
    print('checkpoint_error')
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder1=encoder1,
                                 encoder2=encoder2,
                                 decoder=decoder)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
#print("encoder1.get_weights()",encoder1.get_weights())


# In[69]:



for epoch in range(EPOCHS):
    start = time.time()
    
    hidden1 = encoder1.initialize_hidden_state()
    #hidden2 = encoder2.initialize_hidden_state()
    #user2td = tf.get_variable("user2td",[vocab_user_size,embedding_dim])
    total_loss = 0
    
    for (batch, (user, news, targ)) in enumerate(dataset):
        loss = 0
        #print(user)
        with tf.GradientTape() as tape:
            enc1_output, enc1_hidden = encoder1(news, hidden1)
            
            embedded_user= encoder2(user)
            
            embedded_softmax = embedded_user
            
            embedded_softmax_expand = tf.expand_dims(embedded_softmax,1)
            embedded_softmax_tile = tf.tile(embedded_softmax_expand,[1,max_length_news,1])
            
            enc_output = tf.keras.backend.concatenate([enc1_output,tf.to_float(embedded_softmax_tile)], axis=-1 )

            enc_hidden= tf.keras.backend.concatenate([enc1_hidden,tf.to_float(embedded_softmax)], axis=-1 )
            dec_hidden = enc_hidden
            dec_input = tf.expand_dims([targ_lang.word2idx['<start>']] * BATCH_SIZE, 1)       
            
            # Teacher forcing - feeding the target as the next input
            for t in range(1, targ.shape[1]):
                # passing enc_output to the decoder
                #print(dec_hidden.shape)
                predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
                
                loss += loss_function(targ[:, t], predictions)
                
                # using teacher forcing
                dec_input = tf.expand_dims(targ[:, t], 1)
        
        batch_loss = (loss / int(targ.shape[1]))
        
        total_loss += batch_loss
        
        variables = encoder1.variables + decoder.variables+encoder2.variables
        
        gradients = tape.gradient(loss, variables)
        
        optimizer.apply_gradients(zip(gradients, variables))
        
        if batch % 50 == 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                         batch,
                                                         batch_loss.numpy()))
    # saving (checkpoint) the model every 5 epochs
    if (epoch + 1) % 10 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)
    
    print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                        total_loss / N_BATCH))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

                


