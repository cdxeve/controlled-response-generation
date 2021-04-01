#!/usr/bin/env python
# coding: utf-8


# In[1]:


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
#from tensorflow.python.client import device_lib
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
path_to_file ='data/'+dataset_name+"/question_answer_user.txt"


# Try experimenting with the size of that dataset
num_examples = 10000
target_tensor, news_tensor, targ_lang,news_lang, max_length_targ,max_length_news, = load_dataset(path_to_file, num_examples)

# In[13]:


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




BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE =cong.batch_size
N_BATCH = BUFFER_SIZE//BATCH_SIZE
embedding_dim = 256
style_num=64
units = 512

dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train,news_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)


encoder1 = Encoder(vocab_news_size, embedding_dim, units, BATCH_SIZE)
encoder2=style_model( style_num)
decoder = Decoder(vocab_tar_size, embedding_dim, units+topic_num, BATCH_SIZE)



optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

checkpoint_dir = './_checkpoints/'+dataset_name         
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder1=encoder1,
                                 encoder2=encoder2,
                                 decoder=decoder)

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))



test_num=1000
user_test=input_tensor_val[:test_num]
news_test=news_tensor_val[:test_num]
comment_test=target_tensor_val[:test_num]


# In[1]:


def store_results():
    pred_list=[]
    for i in range(len(news_test)):
        cand=[]
        news=""
        inp1=news_test[i]
        for j in inp1:
            if j!=0:
                news += news_lang.idx2word[j] + ' '
        inp1=tf.convert_to_tensor([inp1])
        inp2=user_test[i]
        inp2=tf.convert_to_tensor([inp2])
        hidden1 = [tf.zeros((1, units))]
        
               enc_output, enc_hidden = encoder1(inp1, hidden1)
        dense_user= encoder2(inp2)


        dec_hidden = enc_hidden
    
        
        dec_input = tf.expand_dims([targ_lang.word2idx['<start>']] , 0)
        for t in range(int(max_length_targ)):
        
            predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_output)
        
            predicted_id = tf.argmax(predictions[0]).numpy()
            print("predictions[0]",predictions[0])
            print("len(predictions[0])",len(predictions[0]))
            cand.append(targ_lang.idx2word[predicted_id])
            if targ_lang.idx2word[predicted_id] == '<end>':
                break
            dec_input = tf.expand_dims([predicted_id], 0)
        
        pred_list.append(cand[:-1])   
    return pred_list

pred_list=store_results()
pred_list=[' '.join(pred) for pred in pred_list]

ref_list=[]
for comment in comment_test:
    ref=[]
    for idx in comment:
        if idx>4:#小于4的是unk,pad，start，end
            ref.append(targ_lang.idx2word[idx])
    ref_list.append(ref)
ref_list=[' '.join(ref) for ref in ref_list]

for i in range(0,2):###好像这里至少要存两遍。。。。#也有可能是忘记close了
    fo_pred= open('results/'+dataset_name+'/pred.txt',encoding='utf8',mode='w')
    fo_ref= open('results/'+dataset_name+'/ref.txt',encoding='utf8',mode='w')
    for line in pred_list:
        if len(line)==0:
            fo_pred.write('ha')
        else:
            fo_pred.write(line)
        fo_pred.write('\n')
    for line in ref_list:
        if len(line)==0:
            fo_ref.write('ha')
        else:
            fo_ref.write(line)
        fo_ref.write('\n')

    i+=1                                




