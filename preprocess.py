from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None
def preprocess_sentence(line):
    dataSet=[]
    line=line.replace('\'','')
    line = re.sub(r"[^a-zA-Z]+", " ", line)
    tokens = word_tokenize(line)  
    #print(tokens)
    tagged_sent = pos_tag(tokens) 

    wnl = WordNetLemmatizer()
    for tag in tagged_sent:
        wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
        dataSet.append(wnl.lemmatize(tag[0], pos=wordnet_pos)) # 词形
   
    w=' '.join(dataSet)
    w = '<start> ' + w + ' <end>'
    return w
   
def preprocess_user(line):
    dataSet=[]
    line=line.replace('\'','')
    line = re.sub(r"[^a-zA-Z]+", " ", line)
    tokens = word_tokenize(line)  
    tagged_sent = pos_tag(tokens)   

    wnl = WordNetLemmatizer()
    for tag in tagged_sent:
        wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
 
    w=' '.join(dataSet)
    return w
# 1. Remove the accents
# 2. Clean the sentences
# 3. Return word pairs in the format: [ENGLISH, SPANISH]
def create_dataset(path, num_examples):
    lines = open(path, encoding='UTF-8').read().strip().split('\n')
    word_pairs = [[ preprocess_sentence(w) for w in re.split('\t|\t ',l)]  for l in lines[:num_examples]]
    
    return word_pairs

class LanguageIndex():
    def __init__(self, lang):
        self.lang = lang
        self.word2idx = {}
        self.idx2word = {}
        self.vocab = set()
        self.create_index()    
    def create_index(self): 
        for phrase in self.lang:
            self.vocab.update(phrase.split(' '))

        self.vocab = sorted(self.vocab)

        self.word2idx['<pad>'] = 0
        self.word2idx['<unk>'] = 1

        for index, word in enumerate(self.vocab):
            self.word2idx[word]=index+2



        for word, index in self.word2idx.items():
            self.idx2word[index]=word
 
def max_length(tensor):
    return max(len(t) for t in tensor)


def load_dataset(path, num_examples):
    # creating cleaned input, output pairs
    pairs = create_dataset(path, num_examples)
    
    # index language using the class defined above    
    #user_lang = LanguageIndex(user for com,news,user in pairs)
    com_lang = LanguageIndex(com for news, com,user in pairs)
    news_lang=LanguageIndex(news for news, com,user in pairs)
    
    # Vectorize the input and target and news languages
    
    # English sentences
    if mode=='backward':
        for news, com,user in pairs:
            temp=com.split()[1:-1]
            temp.reverse()
            com=" ".join(temp)
            com='<start> ' + com + ' <end>'
            print(com)
   
    com_tensor = [[com_lang.word2idx[s] for s in com.split(' ')] for news, com,user in pairs]
    
    #news sentences
    news_tensor=[[news_lang.word2idx[s] for s in news.split(' ')]for news, com,user in pairs]
    # Calculate max_length of input and output and news tensor
    # Here, we'll set those to the longest sentence in the dataset
    max_length_com, max_length_news= max_length(com_tensor), max_length(news_tensor)
    
    # Padding the input and output tand news ensor to the maximum length
    
    com_tensor = tf.keras.preprocessing.sequence.pad_sequences(com_tensor, 
                                                                  maxlen=max_length_com, 
                                                                  padding='post')
    
    news_tensor = tf.keras.preprocessing.sequence.pad_sequences(news_tensor, 
                                                                  maxlen=max_length_news, 
                                                                  padding='post')
    
    return com_tensor, news_tensor,com_lang,news_lang, max_length_com, max_length_news
