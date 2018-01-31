import nltk 
from nltk.stem.lancaster import LancasterStemmer
stemmer=LancasterStemmer()

import numpy as np
import tflearn
import tensorflow as tf
import random
import json
with open('intents.json') as json_data:
   intents=json.load(json_data)

words=[]
classes=[]
documents=[]
ignore_words=['?']
for intent in intents['intents']:
   for pattern in intent['patterns']:
       w=nltk.word_tokenize(pattern)
       words.extend(w)
       documents.append((w,intent['tag']))
       if intent['tag'] not in classes:
           classes.append(intent['tag'])

print (words)


            
