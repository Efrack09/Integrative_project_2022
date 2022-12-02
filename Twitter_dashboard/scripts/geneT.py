import pandas as pd
import numpy as np
import string, os
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
# set seeds for reproducability
#from tensorflow import set_random_seed
import tensorflow as tf
tf.random.set_seed(2) 
from numpy.random import seed
#set_random_seed(2)
seed(1)
# keras module for building LSTM 
#from keras.preprocessing.sequence import pad_sequences
from keras_preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import keras.utils as ku
from keras.utils.data_utils import get_file
import random

from __future__ import print_function
from keras.callbacks import LambdaCallback

from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem import PorterStemmer
import re
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem import PorterStemmer
import re
import sys


class robot_predictor_trainable: 
    
    def __init__(self):
        #here we have the global variables.so far we need 
        #1. the data frames 
        # the model architecture   
        self.df_class_1 = None
        self.df_class_2 = None
        #the tokenizer for each class
        self.tokenizer_class_1 = None
        self.tokenizer_class_2 = None

        #here we need one corpurs per each class
        self.corpus_1 = None
        self.corpus_2 = None
        
        #we need two different tokenizers
        self.tokenizer_1 = Tokenizer()
        self.tokenizer_2 = Tokenizer()
        
        #the models to save
        self.model_1 = None
        self.model_2 = None
        
        #Metadata for trainning class1
        self.total_words_class1=None
        self.predictors_class1=None
        self.label_class1=None
        self.max_sequence_len_class1=None
        
        #Metadata for trainning class2
        self.total_words_class2=None
        self.predictors_class2=None
        self.label_class2=None
        self.max_sequence_len_class2=None
        
        
                
        
    #methods for cleaning the tweets
    #preprocessing methods
    def scrub_words(self,text):
        # remove html markup
        import re
        text=re.sub("(<.*?>)","",text)
    
        #remove non-ascii and digits
        text=re.sub("(\\W|\\d)"," ",text)
  
        #remove whitespace
        text=text.strip()
        return text
        
    #Noise removal, stop word removal, normalizing?
    def cleanString(self,s, special_chars = "\":,.@|ðÿœžðÿâœœïÿœžÿºÿÿœžÿ"):
        from nltk.tokenize import TweetTokenizer
        from nltk.corpus import stopwords



        for char in special_chars:
            s = s.replace(char, "")
        s = s.replace("\n", "")
        s = s.replace("https", "")
        s = self.scrub_words(s)
        tokenizer = TweetTokenizer()
        stop_words = set(stopwords.words('english'))
        cleaned_words = [w for w in tokenizer.tokenize(s) if w not in stop_words]
        return " ".join(cleaned_words)
    
    def cleanFrame(self,frame):
        frame['clean_tweet'] = frame.tweet.apply(self.cleanString)
    

        
        
    
        
    #The methods avaialabe to Load the information  
    def load_dataset_class_1(self):
        df_mexico = pd.read_csv('mexico_en_.csv')
        #lnguage = 'en'
        #df_mexico_1 = df_mexico.loc[df_mexico['lang']==lnguage]        
        t_1=pd.DataFrame({'tweet':list(df_mexico['text'])})
        #load the data to the class
        self.cleanFrame(t_1)
        self.df_class_1 = t_1
        t_1.to_csv('clean_data_c_1.csv')
        print('Data loaded successfully')

    def load_dataset_class_2(self):
        df_argentina = pd.read_csv('argentina_en.csv')
        #lnguage = 'en'
        #df_argentina_1 = df_argentina.loc[df_argentina['lang']==lnguage]        
        t_1=pd.DataFrame({'tweet':list(df_argentina['text'])})
        #load the data to the class
        self.cleanFrame(t_1)
        self.df_class_2 = t_1
        t_1.to_csv('clean_data_c_2.csv')

        print('Data loaded successfully')
        
        
    #=====================================================#
    
    
    #once we have loaded the data we need to generate the corpus per
    #each class, so far the methodology is the next, 
    
    def clean_text(self,txt):
        txt = "".join(t for t in txt if t not in string.punctuation).lower()
        txt = txt.encode("utf8").decode("ascii",'ignore')
        return txt
    
    
    #tokenizer = Tokenizer()
    def get_sequence_of_tokens_1(self,corpus):
        ## tokenization
        self.tokenizer_1.fit_on_texts(corpus)
        total_words = len(self.tokenizer_1.word_index) + 1
    
        ## convert data to a token sequence 
        input_sequences = []
        for line in corpus:
            token_list = self.tokenizer_1.texts_to_sequences([line])[0]
            for i in range(1, len(token_list)):
                n_gram_sequence = token_list[:i+1]
                input_sequences.append(n_gram_sequence)
        return input_sequences, total_words
    
    
    
    def get_sequence_of_tokens_2(self,corpus):
        ## tokenization
        self.tokenizer_2.fit_on_texts(corpus)
        total_words = len(self.tokenizer_2.word_index) + 1
    
        ## convert data to a token sequence 
        input_sequences = []
        for line in corpus:
            token_list = self.tokenizer_2.texts_to_sequences([line])[0]
            for i in range(1, len(token_list)):
                n_gram_sequence = token_list[:i+1]
                input_sequences.append(n_gram_sequence)
        return input_sequences, total_words
    
    
    
    
    
    
    
    def generate_padded_sequences(self,input_sequences,total_words):
        max_sequence_len = max([len(x) for x in input_sequences])
        input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
    
        predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
        label = ku.to_categorical(label, num_classes=total_words)
        return predictors, label, max_sequence_len
    
    #*******************
    def create_model(self,max_sequence_len,total_words):
        input_len = max_sequence_len - 1
        model = Sequential()
        # ----------Add Input Embedding Layer
        model.add(Embedding(total_words,80, input_length=input_len))
        # ----------Add Hidden Layer 1 - LSTM Layer
        model.add(LSTM(700))
        model.add(Dropout(0.4))
        
        # ----------Add Output Layer
        model.add(Dense(total_words, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        return model
    
    #>>>>>>>>>>>>>>>>>>>>>>>><
    #this method is done to create the corpus and train a model in that corpus 
    
    
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    #self.total_words_class1=None
    #self.predictors_class1=None
    #self.label_class1=None
    #self.max_sequence_len_class1=None
    
    def create_corpus_clas_1(self):
        #create the corpus for self.df_class_1
        all_headlines = list(self.df_class_1['clean_tweet'][0:2000])
        corpus = [self.clean_text(x) for x in all_headlines]
        self.corpus_1=corpus
        print(corpus[:10])
        #token created at the beginning
        #tokenizer = Tokenizer()
        #tokenize the corpus
        inp_sequences,self.total_words_class1 = self.get_sequence_of_tokens_1(corpus)
        print(inp_sequences[:10])
        
        #generate padding sequences 
        self.predictors_class1,self.label_class1,self.max_sequence_len_class1= self.generate_padded_sequences(inp_sequences,self.total_words_class1)
        
      
        

        
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    
    
       
    def create_corpus_clas_2(self):
        #create the corpus for self.df_class_1
        all_headlines = list(self.df_class_2['clean_tweet'][0:2000])
        corpus = [self.clean_text(x) for x in all_headlines]
        self.corpus_2=corpus
        print(corpus[:10])
        #token created at the beginning
        #tokenizer = Tokenizer()
        #tokenize the corpus
        inp_sequences,self.total_words_class2 = self.get_sequence_of_tokens_2(corpus)
        print(inp_sequences[:10])
        
        #generate padding sequences 
        self.predictors_class2,self.label_class2,self.max_sequence_len_class2= self.generate_padded_sequences(inp_sequences,self.total_words_class2)
        
      
        
    ##################################################
    #Training 
    def train_class_1(self):
        #use to train 
        model = self.create_model(self.max_sequence_len_class1,self.total_words_class1)
        model.summary()
        
        model.fit(self.predictors_class1,self.label_class1, epochs=20, verbose=5)
        self.model_1 = model
        model.save('Training_class_1.h5')
        
        #Training 
    def train_class_2(self):
        #use to train 
        model = self.create_model(self.max_sequence_len_class2,self.total_words_class2)
        model.summary()
        #print_callback = LambdaCallback(on_epoch_end=self.on_epoch_end)
        model.fit(self.predictors_class2,self.label_class2, epochs=20, verbose=5)
        self.model_2 = model
        model.save('Training_class_2.h5')
        
    
    
    #############################################
    #call back functions
    def sample(self,preds, temperature=1.0):
        # helper function to sample an index from a probability array
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)


    def on_epoch_end(self,epoch, _):
        # Function invoked at end of each epoch. Prints generated text.
        
        
        
        
        
        print()
        print('----- Generating text after Epoch: %d' % epoch)
        text=str(self.corpus_2)
        
        chars = sorted(list(set(text)))
        print('total chars:', len(chars))
        char_indices = dict((c, i) for i, c in enumerate(chars))
        indices_char = dict((i, c) for i, c in enumerate(chars))

        # cut the text in semi-redundant sequences of maxlen characters
        maxlen = 40
        step = 3
        sentences = []
        next_chars = []
        for i in range(0, len(text) - maxlen, step):
            sentences.append(text[i: i + maxlen])
            next_chars.append(text[i + maxlen])
        print('nb sequences:', len(sentences))

        #print('Vectorization...')
        #x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
        #y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
        #for i, sentence in enumerate(sentences):
         #   for t, char in enumerate(sentence):
          #      x[i, t, char_indices[char]] = 1
           # y[i, char_indices[next_chars[i]]] = 1

        
        
        
        
        maxlen=self.max_sequence_len_class2
        start_index = random.randint(0,len(text) - maxlen - 1)
        for diversity in [0.2, 0.5, 1.0, 1.2]:
            print('----- diversity:', diversity)

            generated = ''
            sentence = text[start_index: start_index + maxlen]
            generated += sentence
            print('----- Generating with seed: "' + str(sentence) + '"')
            sys.stdout.write(generated)

            for i in range(400):
                x_pred = np.zeros((1, maxlen, len(chars)))
                for t, char in enumerate(sentence):
                    x_pred[0, t, char_indices[char]] = 1.

                preds = model.predict(x_pred, verbose=0)[0]
                next_index = sample(preds, diversity)
                next_char = indices_char[next_index]

                sentence = sentence[1:] + next_char

                sys.stdout.write(next_char)
                sys.stdout.flush()
            print()


    
    
    
    
    
    
    ###############################################
    #self.total_words_class1=None
    #self.predictors_class1=None
    #self.label_class1=None
    #self.max_sequence_len_class1=None
    #methods to generate Text 
    #here we need to create one tweet of a given class to the other class.
    
    # 1. FROM 1 to 2 
    # 2. FROM 2 to 1
    
    
    def generate_text_from_1_to_1(self,seed_text, next_words):
        #the main workflow is twett_class_1 -> encode_class1-> z -> decode_class_2
        model = tf.keras.models.load_model('Training_class_1.h5')
        gen_w = ""
        
        #enconding_class1 
        for _ in range(next_words):
            token_list = self.tokenizer_1.texts_to_sequences([seed_text])[0]
            token_list = pad_sequences([token_list],maxlen=self.max_sequence_len_class1-1, padding='pre')
            #predicted = model.predict_classes(token_list, verbose=0)
            #decode with model_class_2
            predicted = np.argmax(model.predict(token_list), axis=-1)
            output_word = ""
            #descompress for argentina xd
            for word,index in self.tokenizer_1.word_index.items():
                if index == predicted:
                    output_word = word
                    break
            seed_text += " "+output_word
            gen_w += " "+output_word
            
        return gen_w.title()
    
    

    def generate_text_from_2_to_2(self,seed_text, next_words):
        #the main workflow is twett_class_1 -> encode_class1-> z -> decode_class_2
        model = tf.keras.models.load_model('Training_class_2.h5')
        gen_w = ""
        
        #enconding_class1 
        for _ in range(next_words):
            token_list = self.tokenizer_2.texts_to_sequences([seed_text])[0]
            token_list = pad_sequences([token_list],maxlen=self.max_sequence_len_class2-1, padding='pre')
            #predicted = model.predict_classes(token_list, verbose=0)
            #decode with model_class_2
            predicted = np.argmax(model.predict(token_list), axis=-1)
            output_word = ""
            #descompress for argentina xd
            for word,index in self.tokenizer_2.word_index.items():
                if index == predicted:
                    output_word = word
                    break
            seed_text += " "+output_word
            gen_w += " "+output_word
            
        return gen_w.title()
    