import re
import nltk
import pickle
import streamlit as st
#from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

import gensim
import re
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, classification_report, precision_recall_curve

from sklearn.metrics import accuracy_score

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.layers import LSTM,Dense, Dropout, SpatialDropout1D
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D




#tfidf  = pickle.load(open('vectorizer.pkl', 'rb'))
MODEL_PATH = 'lstm.h5'
model = load_model(MODEL_PATH)
tokenizer  = pickle.load(open('tokenizer.pkl', 'rb'))

################################### Text Preprocessing #####################################

###########                    Removing Punctuations                  ##############
punc = '!#$%&()*+,-./:;<=>?@[\]^_`{|}~।ঃ'
  
def remove_punctuations(text):
    for ele in text:
        if ele in punc:
            text = text.replace(ele, '')

    return text   

###########                    Removing emojis                  ##############

def remove_emojis(text):
    """
    Result :- string without any emojis in it
    Input :- String
    Output :- String
    """
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)

    without_emoji = emoji_pattern.sub(r'',text)
    return without_emoji


#def remove_emoji(text):
   # emoji_pattern = re.compile("["
    #                       u"\U0001F600-\U0001F64F" # emoticons
     #                      u"\U0001F300-\U0001F5FF" # symbols & pictographs
        #                   u"\U0001F680-\U0001F6FF" # transport & map symbols
           #                u"\U0001F1E0-\U0001F1FF" # flags (iOS)
            #               u"\U00002702-\U000027B0"
                #           u"\U000024C2-\U0001F251"
               #            "]+", flags=re.UNICODE)
#    return emoji_pattern.sub(r'', text)

###########                    digits                ##############
def remove_numbers(text):
    """
    Return :- String without numbers
    input :- String
    Output :- String
    """
    number_pattern = r'\d+'
    without_number = re.sub(pattern=number_pattern, repl="", string=text)
    
    return without_number


##################################
##################################

#######################################  REmoving stopwords ##################

stop = {'কি','তুই','ইতি', 'তুঁই', 'তোর','তোয়ারে','বেজ্ঞুনে','আর', 'আঁর', 'নাটক','ওজ্ঞা','ত', 'আঁই', 'ইঁতি', 'ইঁতারে', 'যেই', 
             'ঈতি', 'ঈতারে', 'লই', 'না', 'অইলি', 'দি', 'তইলি', 'তি', 'হন', 'বিয়া', 'যাই', 'হথা','অয়', 'ওরে', 'চাই','কেওর','যদি',
             'মনয়', 'মনত','বেশি', 'নাকি', 'তরারে', 'যেঁত্তে', 'এত্তে', 'হইলেই', 'গেলি', 'ইয়ান', 'নও', 'অইতু', 'কইউম', 'কিছু', 'হইলি',
             'দন', 'চাইতে', 'তুন', 'দে', 'এই', 'ভরি', 'যেন', 'দে', 'অনে', 'কারে', 'লই', 'অইবু', 'মাজখানে', 'দিয়ে', 'গরি', 'নিজর',
             'হইবুয়', 'আগে', 'কাছে', 'আইস', 'তোয়ারে', 'তুনো', 'আছে', 'দিয়ে', 'যা', 'বলে', 'লাগের', 'নাই', 'কেন', 'চাছুনা', 'বলে', 'আইজু',
             'হদ্দে', 'নান', 'আইয়ি', 'মত', 'লাইবু', 'অইল', 'লই', 'সব', 'গরি', 'দিবু', 'কাছেই', 'হর', 'নেকি', 'কস', 'হই', 'মনে','গরে', 'উধু',
             'জাইবু', 'হয়', 'অইলিদি', 'অলর', 'যআইত', 'পারে', 'তুরা', 'নে', 'এত', 'ইতিরে' 'হনে', 'ওদা', 'পরে', 'রহম', 'লাগে', 'ইয়ন',
             'চাইতাম', 'তা', 'যা', 'লগে', 'তে','কেও', 'ন', 'পার', 'ইবা', 'চাই', 'হনো', 'যাই', 'ওই', 'এন', 'চাই', 'সেই' }

#remove_stop_text = []
def text_process(sentence):
        sent = remove_punctuations(sentence)
        sent = re.sub('[a-zA-Z]', '', sent)
        sent = remove_emojis(sent)
        sent = remove_numbers(sent)
        
        #text.append(sent)
        return sent
    
remove_stop_text = [] 
def stop_words(sentences, stop_words):
    word = sentences.split()
    
    for i in word:
        #print(i)
        if i not in stop:
            remove_stop_text.append(i)
    return remove_stop_text
    
    
####################



st.title("Chittagonian Abusive Language Detection")
input_text = st.text_input("Enter the comment")


if st.button('Predict'):
    
    # translating the bangla comment into English
    
    clear_text = text_process(input_text)
    text_without_st = stop_words(clear_text, stop_words)
    
    token = tokenizer.texts_to_sequences([text_without_st])
    print(token)
    padd = pad_sequences(token,maxlen=20)
    print(padd)
    
    pred = model.predict(padd)
    print(pred)
    #prediction = int(model.predict(tw).round().item())
    
    result =  np.where(pred > .5, 1, 0)
    print('perdicted result', result)

    #display

    if result == 1:
        st.header("This is a abusive comment")

    else:
        st.header("This is not a abusive comment")
