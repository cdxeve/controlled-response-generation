#!/usr/bin/env python
# coding: utf-8
from __future__ import absolute_import, division, print_function

# Import TensorFlow >= 1.10 and enable eager execution
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
tf.enable_eager_execution()


from numba import jit
from sklearn.model_selection import train_test_split

import unicodedata
import re
import numpy as np
import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda_device
#print(tf.__version__)
#from tensorflow.python.client import device_lib
#print (device_lib.list_local_devices())
import random

import linecache
import pickle

import warnings
warnings.filterwarnings("ignore")

from preprocess import load_dataset
from model import Encoder,style_model,Decoder,save_list,load_list




# Download the file
dataset_name=config.dataset
path_to_file ='data/'+dataset_name+"/question_answer_user.txt"


# Try experimenting with the size of that dataset
num_examples = 10000
target_tensor, news_tensor, targ_lang,news_lang, max_length_targ,max_length_news, = load_dataset(path_to_file, num_examples,'forward')
r_target_tensor, r_news_tensor,r_targ_lang,r_news_lang, max_length_targ,max_length_news, = load_dataset(path_to_file, num_examples,'backward')

vocab_news_size = len(news_lang.word2idx)
vocab_tar_size = len(targ_lang.word2idx)
print(max_length_targ,max_length_news,vocab_news_size,vocab_tar_size)

list_file = open('data/'+dataset_name+'/w_d_list.pickle','rb')
w_d_list = pickle.load(list_file)

# Creating training and validation sets using an 80-20 split
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val,news_tensor_train,news_tensor_val = train_test_split(w_d_list, target_tensor,news_tensor, test_size=0.2,random_state=1)
r_input_tensor_train, r_input_tensor_val, r_target_tensor_train, r_target_tensor_val,r_news_tensor_train,r_news_tensor_val = train_test_split(w_d_list, r_target_tensor,r_news_tensor, test_size=0.2,random_state=1)


# Show length
len(input_tensor_train), len(target_tensor_train),len(news_tensor_train), len(input_tensor_val), len(target_tensor_val),  len(news_tensor_val)

BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE =cong.batch_size
N_BATCH = BUFFER_SIZE//BATCH_SIZE
embedding_dim = 256
style_num=64
units = 512

encoder1 = Encoder(vocab_news_size, embedding_dim, units, BATCH_SIZE)
encoder2=style_model(style_num)
decoder = Decoder(vocab_tar_size, embedding_dim, units+style_num, BATCH_SIZE)

r_encoder1 = Encoder(vocab_news_size, embedding_dim, units, BATCH_SIZE)
r_encoder2=style_model(style_num)
r_decoder = Decoder(vocab_tar_size, embedding_dim, units+style_num, BATCH_SIZE)


# Define the optimizer and the loss function
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)


checkpoint_dir = './_checkpoints/'+dataset_name         
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder1=encoder1,
                                 encoder2=encoder2,
                                 decoder=decoder)

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


r_checkpoint_dir = './r_checkpoints/'+dataset_name         
r_checkpoint_prefix = os.path.join(r_checkpoint_dir, "ckpt")
r_checkpoint = tf.train.Checkpoint(r_optimizer=optimizer,
                                 r_encoder1=encoder1,
                                 r_encoder2=encoder2,
                                 r_decoder=decoder)

r_checkpoint.restore(tf.train.latest_checkpoint(r_checkpoint_dir))


test_num=1000
user_test=input_tensor_val[:test_num]
news_test=news_tensor_val[:test_num]
comment_test=target_tensor_val[:test_num]
r_user_test=r_input_tensor_val[:test_num]
r_news_test=r_news_tensor_val[:test_num]
r_comment_test=r_target_tensor_val[:test_num]

@jit
def get_enc(index):
    #######forward
    inp1=news_test[index]
    inp1=tf.convert_to_tensor([inp1])

    inp2=user_test[index]
    inp2=tf.convert_to_tensor([inp2])
    
    hidden1 = [tf.zeros((1, units))]
    
    enc_output, enc_hidden = encoder1(inp1, hidden1)
    dense_user= encoder2(inp2)
    
    #######backward
    r_inp1=r_news_test[index]
    r_inp1=tf.convert_to_tensor([r_inp1])

    r_inp2=user_test[index]
    r_inp2=tf.convert_to_tensor([r_inp2])
    
    r_hidden1 = [tf.zeros((1, units))]

    r_enc_output, r_enc_hidden = r_encoder1(r_inp1, r_hidden1)
    r_dense_user= r_encoder2(r_inp2)
    
    return enc_output, enc_hidden,dense_user, r_enc_output,r_enc_hidden,r_dense_user

#######content-contrlled response
from pickle import load

import copy

import sys



def get_list_from_txtfile(fileName):
    dataSet = []
    with open(fileName,encoding='utf-8') as fr:
        for line in fr.readlines():            
            dataSet.append(line.strip())
    return dataSet
# load doc into memory

def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text
import string


input_path='results/'+dataset_name+'/pred.txt'
input_list=get_list_from_txtfile(input_path)
input_list=[preprocess_user(inp) for inp in input_list]
question_list=get_list_from_txtfile('data/'+dataset_name+'/question_test.txt')


from random import randint,choice
max_kw_num=1
kw_list=[]
for i in range(len(input_list)):
    kws=[]
    line=preprocess_user(question_list[i]).split()
    x=randint(1,len(line))
    for j in range(x):
        temp=choice(line)
        if temp in targ_lang.word2idx and temp not in kws and temp.lower() not in input_list[i].lower():
            kws.append(temp)
        if j==max_kw_num-1:
            break
    kw_list.append(kws)

@jit
def delete_duplicate(list_):
        resultList = []
        for item in list_:
            if not item in resultList:
                    resultList.append(item)
        return resultList
@jit
def jaccard(model, reference):
    grams_reference = set(reference)
    grams_model = set(model)
    temp=0
    for i in grams_reference:
        if i in grams_model:
            temp=temp+1
    fenmu=len(grams_model)+len(grams_reference)-temp 
    jaccard_coefficient=float(temp/fenmu)
    return jaccard_coefficient
@jit
def removeDuplicates(input_list):
        removed_list=[]
        for num in input_list:
            if num not in removed_list:
                removed_list.append(num)
        return removed_list    


import heapq
class sentence(object):
    def __init__(self,index,input_list,kw_list,hard_input,hard_constraints=[],
                 p_insert=1/2,p_delete=1/2,
                 phd=9999,
                 preselect_threshold=0.0001, 
                 jaccard_threshold=0,
                 step_times=1):
        self.input_list=removeDuplicates([targ_lang.word2idx[word] for word in input_list.split()])
        
        self.kw_list=[targ_lang.word2idx[word] for word in kw_list]
        self.threshold=preselect_threshold
        self.step_times=step_times
        self.phd=phd
        self.simi=jaccard_threshold
        self.p_insert=p_insert
        self.p_delete=p_delete
        self.hard_constraints=[targ_lang.word2idx[word] for word in hard_constraints]
        self.hard_input=removeDuplicates([targ_lang.word2idx[word] for word in hard_input.split()])
        self.enc_output,self.enc_hidden,self.dense_user,self.r_enc_output,self.r_enc_hidden,self.r_dense_user=get_enc(index)

    
    def _preselect(self,test,decoder):
        selected_list=[]
        if decoder==decoder:
            dec_hidden = self.enc_hidden
        elif decoder==r_decoder:
            dec_hidden=self.r_enc_hidden
        else:
            print('_preselect error')
        if decoder==decoder:
            for t in range(0,len(test)):
                dec_input = tf.expand_dims([test[t]], 0)
                predictions, dec_hidden, beta = decoder(dec_input, dec_hidden, self.enc_output,self.dense_user,t)
        elif decoder==r_decoder:
            for t in range(0,len(test)):
                dec_input = tf.expand_dims([test[t]], 0)
                predictions, dec_hidden, beta = decoder(dec_input, dec_hidden, self.r_enc_output,self.r_dense_user,t)
        else:
            print('_preselect error')
       
        predictions=predictions[0].numpy().tolist()
        
        #print("predictions:",max(predictions))
        selected_list = list(map(predictions.index, heapq.nlargest(10, predictions)))
        #print("selected_word_list",selected_list)
        return selected_list

        
    #@jit
    def preselect(self,added_kw_list,position):  
        #forward
        if position!=0:        
            test=added_kw_list[:position]
            selected_list=self._preselect(test,decoder)          
        #backward
        if position!=len(added_kw_list)-1:                       
            r_test=added_kw_list[position+1:]
            r_test=list(reversed(r_test))
            r_selected_list=self._preselect(r_test,r_decoder)  
            r_selected_list=[r_targ_lang.idx2word[idx] for idx in r_selected_list]
            r_selected_list=[targ_lang.word2idx[word] for word in r_selected_list]
        if position==0 and position==len(added_kw_list)-1:
            return []
        if position==0 and position!=len(added_kw_list)-1:
            return r_selected_list
        if position==len(added_kw_list)-1:
            return selected_list        
        return list(set(selected_list) & set(r_selected_list))
    
    @jit
    def seq_prob(self,encoded):
        
        dec_hidden = self.enc_hidden
        seq_prob=1
        beta_list=[]
        for t in range(0,len(encoded)):
            dec_input = tf.expand_dims([encoded[t]], 0)
            predictions, dec_hidden, beta = decoder(dec_input, dec_hidden, self.enc_output,self.dense_user,t)
            predictions=tf.nn.softmax(predictions)
            word_prob=predictions[0][encoded[t]]
            seq_prob*=word_prob
            beta_list.append(beta)
            #print("seq_prob",seq_prob)
        return seq_prob,beta_list


    def _proposal(self,position,selected_list,added_list,pi_x,operation):
        original_list=copy.deepcopy(added_list)
        pi_list=[]
        for num in selected_list:
            added_list[position]=num
            seq_prob,_=self.seq_prob(added_list)
            pi_list.append(seq_prob)
        s=sum(pi_list)
        if s==0 :
            if operation=='insert' or operation=='append':
                original_list.pop(position)
            return original_list
        g_list=tf.nn.softmax(pi_list)
        
        np.random.seed(random.randint(1, 100))
        p = np.array(g_list)
        new_index = np.random.choice(list(range(len(selected_list))), p = p.ravel())
        
        x=random.uniform(0, 1)
        if operation=='insert' or operation=='append':
            accept_rate=self.p_delete*pi_list[new_index]/(self.p_insert*g_list[new_index]*pi_x)
            if x<min(1,accept_rate):
                added_list[position]=selected_list[new_index]
            else:
                added_list.pop(position)
        
        if operation=='delete':
            pi_x1,_=self.seq_prob(added_list)
            accept_rate=self.p_insert*g_list[new_index]*pi_x1/(self.p_delete*pi_x)
            if x>=min(1,accept_rate):
                return original_list 
            else:
                added_list.pop(position)
        if jaccard(self.hard_input,added_list)<self.simi:
            if operation=='insert' or operation=='append':
                original_list.pop(position)
            return original_list 
       
        return added_list
    
    def proposal(self,added_list,position,operation,):  
        original_list=copy.deepcopy(added_list)
        
        pi_x,_=self.seq_prob(original_list)
        if operation=='insert': 
            added_list.insert(position,self.phd)            
        if operation=='append':
            added_list.append(self.phd)
        if operation=='delete'and (added_list[position] in self.kw_list):
            return original_list 
        if operation=='delete'and (added_list[position] in self.hard_constraints):
            return original_list
        selected_list=self.preselect(added_list,position)

        if any(selected_list):    
            selected_word_list=[targ_lang.idx2word[x] for x in selected_list]
            #print("selected_word_list",selected_word_list)
            new_list=self._proposal(position,selected_list,added_list,pi_x,operation)
            
            return new_list
        else:
            #print('kong','insert_reject')
            return original_list
    
    
    #@jit
    def modify(self):
        added_kw_list=[self.input_list]
        for kw in self.kw_list:
            new=[]
            for sentence in added_kw_list:
                bubian=copy.deepcopy(sentence)
                _,beta_list=self.seq_prob(bubian)
                position_list=list(map(beta_list.index, heapq.nsmallest(max(1,len(bubian)//2), beta_list)))
                
                for position in position_list:                
                    sentence.pop(position)
                    sentence.insert(position,kw)
                    new.append(sentence)
                    sentence=copy.deepcopy(bubian)
                    
            added_kw_list=new
        modified_list=[]
        modified_list.extend(added_kw_list)
        step=0                      
        while step<self.step_times:
            step+=1
            print("step:",step)
            new_list=[]
            for added_kw in added_kw_list:
                bubian_list=copy.deepcopy(added_kw)                    
                for position in range(len(added_kw)):
                    np.random.seed(random.randint(1, 100))
                    
                    p = np.array([self.p_insert, self.p_delete])
                    operation = np.random.choice(['insert', 'delete'], p = p.ravel())
                    
                    added_list=copy.deepcopy(bubian_list)
                    print('#######added_list:',[targ_lang.idx2word[num] for num in added_list],position)
                    new_list.append(self.proposal(added_list,position,operation))
                added_list=copy.deepcopy(bubian_list)
                position=len(added_list)
                operation = np.random.choice(['append', 'delete'], p = p.ravel())

                if operation=='append':
                    new_list.append(self.proposal(added_list,position,operation))
                
                new_list=delete_duplicate(new_list)
                seq_list=[]
                for new in new_list:
                    seq_list.append([targ_lang.idx2word[idx] for idx in new])
            modified_list.extend(new_list)
            added_kw_list=new_list
        modified_list=delete_duplicate(modified_list)
        
        print('modified_list')
        for modified in modified_list:
            print([targ_lang.idx2word[idx] for idx in modified])
            
        return modified_list

    
    def decide(self):
        first_prob,_=self.seq_prob(self.input_list)
        if len(self.input_list)>0:
            first_ppl=pow(first_prob,-1/len(self.input_list))
        else:
            first_ppl=self.phd
        first_sentence=[targ_lang.idx2word[idx] for idx in self.input_list]

        modified_list=self.modify()
        ppl_list=[]
        prob_list=[]
        for sentence in modified_list:
            prob,_=self.seq_prob(sentence)
            prob_list.append(prob)
            if len(sentence)>0:
                ppl=pow(prob,-1/len(sentence))
            else:
                ppl=self.phd
            ppl_list.append(ppl)
            min_index=ppl_list.index(min(ppl_list))
            max_index=prob_list.index(max(prob_list))
          
        final=[targ_lang.idx2word[idx] for idx in modified_list[min_index]]  
        print('first round：',first_sentence,'ppl：',first_ppl)
        print('final round：',final,'ppl：',min(ppl_list))  
        return " ".join(final)
            
@jit
def one_example_test(index,input_list,kw_list,turn_num):
    print('###############round 0#######')
    result_list=[]
    print(input_list,kw_list)
    john=sentence(index,input_list,kw_list,input_list)
    hard_constraints=kw_list
    result_list.append(john.decide())      
    for i in range(1,turn_num):
        print('################the',i,'round#############')
        print(result_list[i-1])
        x=sentence(index,result_list[i-1],[],input_list,hard_constraints)       
        result_list.append(x.decide())
    return result_list[-1]

@jit
def save_doc(lines, filename):
  
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()
    
@jit
def test(start,end,turn_num=1):
    outFile='results/'+dataset_name+'/pred_2.txt'
    if(os.path.exists(outFile)):
            os.remove(outFile)
            print ('removed')
    else:
            print('not existed')
    lists=[]
    for i in range(start,end):
        fo= open(outFile,encoding='utf8',mode='a')
        print(i+1,'sentence')
        final=one_example_test(i,input_list[i],kw_list[i],turn_num)
        print('final',final)
        #lists.append(one_example_test(i,input_list[i],kw_list[i],turn_num))
        fo.write(final)
        fo.write('\n')
        fo.close()
    return lists

test(config.evaluate_start,config.evaluate_end)
