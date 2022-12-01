import pandas as pd
import seaborn as sns
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem import PorterStemmer
import re
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing.label import MultiLabelBinarizer
from sklearn.preprocessing._label import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer

import tensorflow as tf

import numpy as np 
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import Dropout

class robot_class_predictor: 
    def __init__(self):
        #here we have the global variables.so far we need 
        #1. the data frames 
        # the model architecture   
        #Noise removal, stop word removal, normalizing?
        self.vectorizer = None
        self.model = None
        self.df_class_1=None
        self.df_class_2=None
        self.concat_df=None
        self.tfidf=None
        
        
        
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

        
        
    
        
    #The methods avaialabe to Load the information  
    def load_dataset_class_1(self):
        df_mexico = pd.read_csv('../NLP_notebooks/classificator/clean_data_c_1.csv')
        lnguage = 'en'
        df_mexico_1 = df_mexico.loc[df_mexico['lang']==lnguage]        
        t_1=pd.DataFrame({'tweet':list(df_mexico_1['text']),'class':0})
        #load the data to the class
        self.cleanFrame(t_1)
        self.df_class_1 = t_1

        print('Data loaded successfully')

    def load_dataset_class_2(self):
        df_argentina = pd.read_csv('../NLP_notebooks/classificator/clean_data_c_2.csv')
        lnguage = 'en'
        df_argentina_1 = df_argentina.loc[df_argentina['lang']==lnguage]        
        t_1=pd.DataFrame({'tweet':list(df_argentina_1['text']),'class':1})
        #load the data to the class
        self.cleanFrame(t_1)
        self.df_class_2 = t_1


        print('Data loaded successfully')
        
        
    #=====================================================#
    def tweetToVec(self,tweet, row=0):
        words = tweet.split(" ")
        vec = np.zeros(self.tfidf.shape[1])
        for w in words:
            #print("including word " + w)
            if w in self.tfidf.columns:
                index = self.tfidf.columns.get_loc(w)
            #print(index, tfidf[w][row])
            vec[index] = self.tfidf[w][row]
        return vec
    
    
    def prepare_metadata(self):
        self.load_dataset_class_1()
        self.load_dataset_class_2()
        
        self.concat_df = pd.concat([self.df_class_1,self.df_class_2])
        
            #from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer()
        vectors = self.vectorizer.fit_transform(self.concat_df.tweet.tolist())
        feature_names = self.vectorizer.get_feature_names_out()
        dense = vectors.todense()
        denselist = dense.tolist()
        self.tfidf = pd.DataFrame(denselist, columns=feature_names)
        self.tfidf.head()
        
        
        
        ############################3
    def predict_new(self,tweet):
        model = tf.keras.models.load_model('../NLP_notebooks/classificator/Training_clasification_v1.h5')
        #vectorize = self.tweetToVec(tweet,row=0)
        tweet2 = [tweet,'a']
        vectors = self.vectorizer.transform(tweet2)
        feature_names = self.vectorizer.get_feature_names()
        dense = vectors.todense()
        denselist = dense.tolist()
        tfidf = pd.DataFrame(denselist, columns=feature_names)
        X=tfidf.values
        
        #X = vectorize
        #Adding bias term
        X = np.c_[np.ones(X.shape[0]), X]


        val = model.predict(X).round()
        print(val)

        return val

wall_e = robot_class_predictor()
#wall_e.prepare_metadata()
test =  'Lamediocre #SeleccionMexicana to the #Brazil National Team already in the round of 16. https://t.co/rhisRaKL2h'
ect = wall_e.predict_new(test)
ect[0]