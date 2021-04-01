import tensorflow.compat.v1 as tf
import pickle

tf.disable_v2_behavior()
tf.enable_eager_execution()

def save_list(list_name,path_str):
    list_file=open(str(path_str),'wb')
    pickle.dump(list_name,list_file)
    list_file.close()
def load_list(path_str):
    list_file=open(str(path_str),'rb')
    list_name=pickle.load(list_file)
    return list_name
def gru(units):
  # If you have a GPU, we recommend using CuDNNGRU(provides a 3x speedup than GRU)
  # the code automatically does that.
    if tf.test.is_gpu_available():
        return tf.keras.layers.CuDNNGRU(units, 
                                    return_sequences=True, 
                                    return_state=True, 
                                    recurrent_initializer='glorot_uniform')
    else:
        return tf.keras.layers.GRU(units, 
                               return_sequences=True, 
                               return_state=True, 
                               recurrent_activation='sigmoid', 
                               recurrent_initializer='glorot_uniform')

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = gru(self.enc_units)
        
    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state = hidden)        
        return output, state
    
    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))

class style_model(tf.keras.Model):
    def __init__(self, topic_num):
        super(style_model, self).__init__()
        self.dense=tf.keras.layers.Dense(topic_num,bias_initializer=tf.keras.initializers.constant(0.0), dtype='float64')
    def call(self, x):
        #dense_user.shape ==(batch_size, style_num)
        dense_user = self.dense(x)    
        return dense_user

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = gru(self.dec_units)
        self.fc = tf.keras.layers.Dense(vocab_size)
        self.vocab_sz=vocab_size
        # used for attention
        self.W1 = tf.keras.layers.Dense(self.dec_units)
        self.W2 = tf.keras.layers.Dense(self.dec_units)
        self.V = tf.keras.layers.Dense(1)
        
    def call(self, x, hidden, enc_output,dense_user,t):
        # enc_output shape == (batch_size, max_length, hidden_size)
        
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        
        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying tanh(FC(EO) + FC(H)) to self.V
        score = self.V(tf.nn.tanh(self.W1(enc_output) + self.W2(hidden_with_time_axis)))
        
        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)
        
        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * enc_output
        context_vector = tf.reduce_sum(context_vector, axis=1)
        
        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)
        
        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        
        # passing the concatenated vector to the GRU
        output, state = self.gru(x)
        
        ###########################################adaptive cue gate################################
        #state shape == (batch_size,hidden_size)
        #enc_output shape ==(batch_size, max_length, hidden_size)
        #context_vector shape ==(batch_size, hidden_size)
        
        concate=tf.concat([state,enc_output[:,:,t],context_vector],axis=1)
        
        #concate shape == (batch_size, 1+max_length+1, hidden_size)
        
        #MLP
        #MLP=tf.keras.layers.Denise(self.dec_units)(concate)
        beta=tf.keras.layers.Dense(1,activation='sigmoid')(concate) 
        #beta shape == (batch_size, 1)
        #print("beta.shape:",beta)
        
        #dense_user.shape ==(batch_size, topic_num)
        alpha=beta*dense_user   
        #alpha.shape ==(batch_size, topic_num)
        
        output=tf.concat([state,context_vector,alpha],axis=1)
        # output shape == (batch_size * 1, hidden_size*2+topic_num)
        
        #output = tf.reshape(output, (-1, output.shape[2]))
        
        # output shape == (batch_size * 1, vocab)
        x = self.fc(output)
        
        return x, state, beta
        
    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.dec_units))
