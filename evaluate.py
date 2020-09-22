#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8


# In[1]:


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




# In[2]:


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



# In[13]:


vocab_news_size = len(news_lang.word2idx)
vocab_tar_size = len(targ_lang.word2idx)
print(max_length_targ,max_length_news,vocab_news_size,vocab_tar_size)


# In[14]:

#读取
list_file = open('../style-RG/w_d/'+dataset_name+'/w_d_list.pickle','rb')
w_d_list = pickle.load(list_file)


# In[10]:


# Creating training and validation sets using an 80-20 split
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val,news_tensor_train,news_tensor_val = train_test_split(w_d_list, target_tensor,news_tensor, test_size=0.2,random_state=1)
r_input_tensor_train, r_input_tensor_val, r_target_tensor_train, r_target_tensor_val,r_news_tensor_train,r_news_tensor_val = train_test_split(w_d_list, r_target_tensor,r_news_tensor, test_size=0.2,random_state=1)


# Show length
len(input_tensor_train), len(target_tensor_train),len(news_tensor_train), len(input_tensor_val), len(target_tensor_val),  len(news_tensor_val)


# In[17]:


# In[4]:



BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE =cong.batch_size
N_BATCH = BUFFER_SIZE//BATCH_SIZE
embedding_dim = 256
topic_num=64
units = 512

encoder1 = Encoder(vocab_news_size, embedding_dim, units, BATCH_SIZE)
encoder2=style_model( topic_num)
decoder = Decoder(vocab_tar_size, embedding_dim, units+topic_num, BATCH_SIZE)

r_encoder1 = Encoder(vocab_news_size, embedding_dim, units, BATCH_SIZE)
r_encoder2=style_model( topic_num)
r_decoder = Decoder(vocab_tar_size, embedding_dim, units+topic_num, BATCH_SIZE)


# ## Define the optimizer and the loss function

# In[66]:



optimizer = tf.train.AdamOptimizer(learning_rate=0.001)


#def loss_function(real, pred):
 # mask = 1 - np.equal(real, 0)
  #loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred) * mask
  #return tf.reduce_mean(loss_)


# ## Checkpoints (Object-based saving)

# In[67]:


checkpoint_dir = '../style-RG/checkpoints/'+dataset_name         
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder1=encoder1,
                                 encoder2=encoder2,
                                 decoder=decoder)

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


r_checkpoint_dir = '../style-RG/r_checkpoints/'+dataset_name         
r_checkpoint_prefix = os.path.join(r_checkpoint_dir, "ckpt")
r_checkpoint = tf.train.Checkpoint(r_optimizer=optimizer,
                                 r_encoder1=encoder1,
                                 r_encoder2=encoder2,
                                 r_decoder=decoder)

r_checkpoint.restore(tf.train.latest_checkpoint(r_checkpoint_dir))

#print("encoder1.get_weights(),r_encoder1.get_weights",encoder1.get_weights(),r_encoder1.get_weights())

test_num=1000
user_test=input_tensor_val[:test_num]
news_test=news_tensor_val[:test_num]
comment_test=target_tensor_val[:test_num]
r_user_test=r_input_tensor_val[:test_num]
r_news_test=r_news_tensor_val[:test_num]
r_comment_test=r_target_tensor_val[:test_num]


# In[1]:


@jit
def get_enc(index):
    #######正向部分
    inp1=news_test[index]
    inp1=tf.convert_to_tensor([inp1])

    inp2=user_test[index]
    inp2=tf.convert_to_tensor([inp2])
    
    hidden1 = [tf.zeros((1, units))]
    
    enc1_output, enc1_hidden = encoder1(inp1, hidden1)
    embedded_user= encoder2(inp2)
    embedded_softmax = embedded_user
    embedded_softmax_expand = tf.expand_dims(embedded_softmax,1)
    embedded_softmax_tile = tf.tile(embedded_softmax_expand,[1,max_length_news,1])

    enc_output = tf.keras.backend.concatenate([enc1_output,tf.to_float(embedded_softmax_tile)], axis=-1 )
    enc_hidden= tf.keras.backend.concatenate([enc1_hidden,tf.to_float(embedded_softmax)], axis=-1 )
    #######逆向部分
    r_inp1=r_news_test[index]
    r_inp1=tf.convert_to_tensor([r_inp1])

    r_inp2=user_test[index]
    r_inp2=tf.convert_to_tensor([r_inp2])
    
    r_hidden1 = [tf.zeros((1, units))]

    r_enc1_output, r_enc1_hidden = r_encoder1(r_inp1, r_hidden1)
    r_embedded_user= r_encoder2(r_inp2)
    r_embedded_softmax = r_embedded_user
    r_embedded_softmax_expand = tf.expand_dims(r_embedded_softmax,1)
    r_embedded_softmax_tile = tf.tile(r_embedded_softmax_expand,[1,max_length_news,1])

    r_enc_output = tf.keras.backend.concatenate([r_enc1_output,tf.to_float(r_embedded_softmax_tile)], axis=-1 )

    r_enc_hidden= tf.keras.backend.concatenate([r_enc1_hidden,tf.to_float(r_embedded_softmax)], axis=-1 )
    #print("encoder1.get_weights(),r_encoder1.get_weights",encoder1.get_weights(),r_encoder1.get_weights())

    
    return enc_output, enc_hidden, r_enc_output,r_enc_hidden








# In[7]:


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


input_path='../style-RG/results/'+dataset_name+'/pred.txt'
input_list=get_list_from_txtfile(input_path)
input_list=[preprocess_user(inp) for inp in input_list]
question_list=get_list_from_txtfile('data/'+dataset_name+'/question_test.txt')


# In[39]:


from random import randint,choice
max_kw_num=1
kw_list=[]#双层列表
for i in range(len(input_list)):
    kws=[]#单层列表
    line=preprocess_user(question_list[i]).split()#随机选定一个句子
    x=randint(1,len(line))#随机指定抽取的关键词数
    for j in range(x):
        temp=choice(line)#随机选定一个单词
        #if temp not in kws:#保证没有被选过
        if temp in targ_lang.word2idx and temp not in kws and temp.lower() not in input_list[i].lower():#保证不在input并且没有被选过
            kws.append(temp)
        if j==max_kw_num-1:
            break#保证关键词数不过多
    #if len(kws)<1:
     #   kws.append("yes")
    kw_list.append(kws)
print(kw_list[:10],input_list[:10])



# In[8]:


# In[40]:

@jit
def delete_duplicate(list_):
        resultList = []
        for item in list_:
            if not item in resultList:
                    resultList.append(item)
        return resultList
@jit
def jaccard(model, reference):#terms_reference为源句子，terms_model为候选句子
    grams_reference = set(reference)#去重；如果不需要就改为list
    grams_model = set(model)
    temp=0
    for i in grams_reference:
        if i in grams_model:
            temp=temp+1
    fenmu=len(grams_model)+len(grams_reference)-temp #并集
    jaccard_coefficient=float(temp/fenmu)#交集
    return jaccard_coefficient
@jit
def removeDuplicates(input_list):#用于删除单个句子中的重复信息
        removed_list=[]
        for num in input_list:
            if num not in removed_list:
                removed_list.append(num)
        return removed_list    


# In[25]:


import heapq
class sentence(object):
    def __init__(self,index,input_list,kw_list,hard_input,hard_constraints=[],
                 p_insert=1/2,p_delete=1/2,
                 phd=9999,
                 preselect_threshold=0.0001, 
                 jaccard_threshold=0,
                 step_times=1):
        self.input_list=removeDuplicates([targ_lang.word2idx[word] for word in input_list.split()])
        ##试试不要原句直接生成
        #self.input_list=[]
        self.kw_list=[targ_lang.word2idx[word] for word in kw_list]
        self.threshold=preselect_threshold
        self.step_times=step_times
        self.phd=phd
        self.simi=jaccard_threshold
        self.p_insert=p_insert
        self.p_delete=p_delete
        ######两个hard变量在后几轮中保证原句和关键词
        self.hard_constraints=[targ_lang.word2idx[word] for word in hard_constraints]
        self.hard_input=removeDuplicates([targ_lang.word2idx[word] for word in hard_input.split()])
        #print(self.hard_constraints)
        self.enc_output,self.enc_hidden,self.r_enc_output,self.r_enc_hidden=get_enc(index)

    
    def _preselect(self,test,decoder):
        selected_list=[]
        if decoder==decoder:
            dec_hidden = self.enc_hidden
        elif decoder==r_decoder:
            dec_hidden=self.r_enc_hidden
        else:
            print('_preselect error')
        #test.insert(0,targ_lang.word2idx['<start>'])
        #test.append(targ_lang.word2idx['<end>'])
        if decoder==decoder:
            for t in range(0,len(test)):
                dec_input = tf.expand_dims([test[t]], 0)
                predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, self.enc_output)
        elif decoder==r_decoder:
            for t in range(0,len(test)):
                dec_input = tf.expand_dims([test[t]], 0)
                predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, self.r_enc_output)
        else:
            print('_preselect error')
        #predictions=tf.nn.softmax(predictions)
        #prob_list=predictions[0]
        #print(max(prob_list))
        ###############把threshold的范围改了
       # threshold=max(prob_list)
        #for i in range(len(prob_list)):
            #if prob_list[i]>=self.threshold:
                #selected_list.append(i)
        #直接得到最大的三个单词的索引
        predictions=predictions[0].numpy().tolist()
        #print("predictions:",max(predictions))
        selected_list = list(map(predictions.index, heapq.nlargest(10, predictions)))
        #print("selected_word_list",selected_list)
        return selected_list

        
    #@jit
    def preselect(self,added_kw_list,position):  
        #print('preselect...')
        #前向预选
        if position!=0:        
            test=added_kw_list[:position]
            #print('前向预选。。。',test)
            selected_list=self._preselect(test,decoder)          
        #后向预选
        if position!=len(added_kw_list)-1:                       
            r_test=added_kw_list[position+1:]
            r_test=list(reversed(r_test))
            #print('后向预选。。。',r_test)
            r_selected_list=self._preselect(r_test,r_decoder)  
            r_selected_list=[r_targ_lang.idx2word[idx] for idx in r_selected_list]
            r_selected_list=[targ_lang.word2idx[word] for word in r_selected_list]#转成同一种表示方式
        if position==0 and position==len(added_kw_list)-1:
            #print('预选时遇到空列表')
            return []
        if position==0 and position!=len(added_kw_list)-1:
            #print(len(added_kw_list)-1,position)
            return r_selected_list
        if position==len(added_kw_list)-1:
            return selected_list        
        return list(set(selected_list) & set(r_selected_list))
    
    @jit
    def seq_prob(self,encoded):
        
        dec_hidden = self.enc_hidden
        #encoded.insert(0,targ_lang.word2idx['<start>'])
        #encoded.append(targ_lang.word2idx['<end>'])
        seq_prob=1
        for t in range(0,len(encoded)):
            dec_input = tf.expand_dims([encoded[t]], 0)
            predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, self.enc_output)
            predictions=tf.nn.softmax(predictions)
            word_prob=predictions[0][encoded[t]]
            seq_prob*=word_prob
            #print("seq_prob",seq_prob)
        return seq_prob


    def _proposal(self,position,selected_list,added_list,pi_x,operation):
        #print('_proposal')
        original_list=copy.deepcopy(added_list)
        pi_list=[]
        for num in selected_list:
            added_list[position]=num
            pi_list.append(self.seq_prob(added_list))
        s=sum(pi_list)
        if s==0 :
           # print('分母为0')
            #####insert和append两种情况在进入——proposal前已经加了占位符
            if operation=='insert' or operation=='append':
            #    print('insert_reject')
                original_list.pop(position)
            return original_list
        #g_insert的概率
        g_list=tf.nn.softmax(pi_list)
        #g_list=[pi/s for pi in pi_list]
        
        np.random.seed(random.randint(1, 100))
        p = np.array(g_list)
        #print("p:",p) 
        #print('sump:',sum(p))
        ###判断有没有nan
        #for i in range(len(p)):
            
         #   if np.isnan(p[i]):
          #      p[i]=0
                #selected_list.pop(i)
           #     print('有nan')
        new_index = np.random.choice(list(range(len(selected_list))), p = p.ravel())
        
        x=random.uniform(0, 1)
        if operation=='insert' or operation=='append':
            #print(operation,[targ_lang.idx2word[idx] for idx in added_list])


            accept_rate=self.p_delete*pi_list[new_index]/(self.p_insert*g_list[new_index]*pi_x)
            #print('insert的接受率：',min(1,accept_rate))
            if x<min(1,accept_rate):
                #print('insert_accept')
                added_list[position]=selected_list[new_index]
                #print([targ_lang.idx2word[idx] for idx in added_list])
            else:
                #print('insert_reject')
                added_list.pop(position)
        
        if operation=='delete':
            #print(operation)

           # print([targ_lang.idx2word[idx] for idx in added_list])
            
            pi_x1=self.seq_prob(added_list)
            #accept_rate=self.p_insert*pi_x1/(self.p_delete*(pi_x+s))
            accept_rate=self.p_insert*g_list[new_index]*pi_x1/(self.p_delete*pi_x)
            if x>=min(1,accept_rate):
                #print('delete_reject')
                return original_list 
            else:
                #print('delete_accept')
                added_list.pop(position)
                #print([targ_lang.idx2word[idx] for idx in added_list])
        if jaccard(self.hard_input,added_list)<self.simi:
            if operation=='insert' or operation=='append':
                original_list.pop(position)
            return original_list 
       
        return added_list
    
    #@jit
    def proposal(self,added_list,position,operation,):  
        #有关键词但没有被修改的句子
        original_list=copy.deepcopy(added_list)
        
        pi_x=self.seq_prob(original_list)
        #print('有关键词但没有被修改的句子概率：',pi_x,'位置',position)
        if operation=='insert': 
            #先插入占位符phd            
            added_list.insert(position,self.phd)            
        if operation=='append':
            added_list.append(self.phd)
        if operation=='delete'and (added_list[position] in self.kw_list):
            #print('delete_reject')
            return original_list 
        #hard_constraints保证后几轮修改时，第0轮的关键词不会被删除
        if operation=='delete'and (added_list[position] in self.hard_constraints):
            #print('delete_reject')
            return original_list
        #进行预挑选
        selected_list=self.preselect(added_list,position)


        #判断预挑选结果是否为空
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
        #第一轮添加所有关键词 
        print("添加关键词。。")
        added_kw_list=[self.input_list]
        for kw in self.kw_list:
            new=[]
            for sentence in added_kw_list:
                #不变一定要指两次
                bubian=copy.deepcopy(sentence)
                ####给每一个句子在不同的位置添加关键词
                for i in range(len(sentence)+1):                
                    #print("被添加位置：",i,'关键词：',kw,tokenizer.index_word[kw])               
                    if i<len(sentence):
                        sentence.insert(i,kw)
                        new.append(sentence)
                    else:
                        sentence.append(kw)
                        new.append(sentence)
                    sentence=copy.deepcopy(bubian)
                    
                    #添加了关键词的list.
            added_kw_list=new
        #print(added_kw_list)
        #第二轮对添加后的进行修改
        print("语法调整。。。") 
        modified_list=[]
        modified_list.extend(added_kw_list)
        step=0                      
        #开始循环调整
        while step<self.step_times:
            step+=1
            print("step:",step)
            new_list=[]
            #print('added_kw_list',tokenizer.sequences_to_texts(added_kw_list))
            for added_kw in added_kw_list:
                bubian_list=copy.deepcopy(added_kw)                    
                #接下来对这个句子进行语法调整                              
                for position in range(len(added_kw)):
                    #依概率抽取一个proposal 
                    np.random.seed(random.randint(1, 100))
                    
                    p = np.array([self.p_insert, self.p_delete])
                    operation = np.random.choice(['insert', 'delete'], p = p.ravel())
                    #if position ==len(added_kw)-1:
                    #print("操作：", operation)
                    #进行操作
                    added_list=copy.deepcopy(bubian_list)
                    print('#######added_list:',[targ_lang.idx2word[num] for num in added_list],position)
                    new_list.append(self.proposal(added_list,position,operation))
                    #print(tokenizer.sequences_to_texts(modified_list))
                #到结尾处只能append
                added_list=copy.deepcopy(bubian_list)
                position=len(added_list)
                #print('#######added_kw_list',added_list,position)
                operation = np.random.choice(['append', 'delete'], p = p.ravel())

                if operation=='append':
                
                #print("操作：", operation)
                    new_list.append(self.proposal(added_list,position,operation))
                
                new_list=delete_duplicate(new_list)
                seq_list=[]
                for new in new_list:
                    seq_list.append([targ_lang.idx2word[idx] for idx in new])
                #print('new_list', seq_list)           
            modified_list.extend(new_list)
            added_kw_list=new_list
        modified_list=delete_duplicate(modified_list)
        
        print('modified_list')
        for modified in modified_list:
            print([targ_lang.idx2word[idx] for idx in modified])
        #print(tokenizer.sequences_to_texts(modified_list))
            
        return modified_list

    
    def decide(self):
        #先保存第一个例子
        first_prob=self.seq_prob(self.input_list)
            #print(first_prob,len(self.input_list))
        if len(self.input_list)>0:
            first_ppl=pow(first_prob,-1/len(self.input_list))
        else:
            first_ppl=self.phd
        first_sentence=[targ_lang.idx2word[idx] for idx in self.input_list]

        modified_list=self.modify()
        ppl_list=[]
        prob_list=[]
        for sentence in modified_list:
            #print(sentence)
            
            prob=self.seq_prob(sentence)
            prob_list.append(prob)
            if len(sentence)>0:
                ppl=pow(prob,-1/len(sentence))
            else:
                ppl=self.phd
            ppl_list.append(ppl)
            min_index=ppl_list.index(min(ppl_list))
            max_index=prob_list.index(max(prob_list))
          
        final=[targ_lang.idx2word[idx] for idx in modified_list[min_index]]
        #final=tokenizer.sequences_to_texts([modified_list[max_index]])
        #print('前一轮结果：',first_sentence,'ppl：',first_ppl,'句子概率：',first_prob)
        #print('最终结果：',final,'ppl：',min(ppl_list),'句子概率：',prob_list[min_index])    
        print('前一轮结果：',first_sentence,'ppl：',first_ppl)
        print('最终结果：',final,'ppl：',min(ppl_list))  
        return " ".join(final)
            


# In[26]:


@jit
def one_example_test(index,input_list,kw_list,turn_num):
    print('###############第0轮#######')
    #result_list中存有 【句子，困惑度，句子概率】
    result_list=[]
    #第0轮，有kw
    print(input_list,kw_list)
    john=sentence(index,input_list,kw_list,input_list)
    hard_constraints=kw_list
    result_list.append(john.decide())      
    #从弟1轮开始就没有kw了
    for i in range(1,turn_num):
        print('################第',i,'轮#############')
        print(result_list[i-1])
        x=sentence(index,result_list[i-1],[],input_list,hard_constraints)       
        result_list.append(x.decide())
    return result_list[0]

@jit
def save_doc(lines, filename):
    
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()
    
@jit
def test(start,end,turn_num=1):
    lists=[]
    for i in range(start,end):
        print('第',i+1,'个句子')
        lists.append(one_example_test(i,input_list[i],kw_list[i],turn_num))
    print(lists)
    save_doc(lists,'results/'+dataset_name+'/pred_1.txt')
    return lists

test(config.evaluate_start,config.evaluate_end)


