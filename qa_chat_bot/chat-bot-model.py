# -*- coding: utf-8 -*-
import nltk 
from nltk.stem.lancaster import LancasterStemmer
stemmer=LancasterStemmer()
import os
import numpy as np
import tflearn
import tensorflow as tf
import random
import json
from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers import Input, Activation, Dense, Permute, Dropout
from keras.layers import add, dot, concatenate
from keras.layers import LSTM
from keras.utils.data_utils import get_file
from keras.preprocessing.sequence import pad_sequences
from functools import reduce
import tarfile
import re

#https://github.com/brilee/MuGo
"""
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

#print (words)
print (documents)
print(classes)
"""
"""
try:
    path = get_file('babi-tasks-v1-2.tar.gz', origin='https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz')
except:
    print('Error downloading dataset, please download it manually:\n'
          '$ wget http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz\n'
          '$ mv tasks_1-20_v1-2.tar.gz ~/.keras/datasets/babi-tasks-v1-2.tar.gz')
    raise
"""
def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]


def parse_stories(lines, only_supporting=False):
    '''Parse stories provided in the bAbi tasks format
    If only_supporting is true, only the sentences
    that support the answer are kept.
    '''
    data = []
    story = []
    for line in lines:
        line = line.decode('utf-8').strip()
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line:
            q, a, supporting = line.split('\t')
            q = tokenize(q)
            substory = None
            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                substory = [x for x in story if x]
            data.append((substory, q, a))
            story.append('')
        else:
            sent = tokenize(line)
            story.append(sent)
    return data


def get_stories(f, only_supporting=False, max_length=None):
    '''Given a file name, read the file,
    retrieve the stories,
    and then convert the sentences into a single story.
    If max_length is supplied,
    any stories longer than max_length tokens will be discarded.
    '''
    data = parse_stories(f.readlines(), only_supporting=only_supporting)
    flatten = lambda data: reduce(lambda x, y: x + y, data)
    data = [(flatten(story), q, answer) for story, q, answer in data if not max_length or len(flatten(story)) < max_length]
    return data

def getData(path,s=False):
 
    challenges = {
        # QA1 with 10,000 samples
        'single_supporting_fact_10k': 'tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_{}.txt',
        # QA2 with 10,000 samples
        'two_supporting_facts_10k': 'tasks_1-20_v1-2/en-10k/qa2_two-supporting-facts_{}.txt',
    }
    challenge_type = 'single_supporting_fact_10k'
    challenge = challenges[challenge_type]
    if s:    
        path = get_file('babi-tasks-v1-2.tar.gz', origin='https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz')
        tar = tarfile.open(path)
        train_stories = get_stories(tar.extractfile(challenge.format('train')))
        test_stories = get_stories(tar.extractfile(challenge.format('test')))
    else:
        train_stories = get_stories(open(os.path.join(path,challenge.format('train'))))
        test_stories = get_stories(open(os.path.join(path,challenge.format('test'))))
   
    print('Extracting stories for the challenge:', challenge_type)
    return train_stories,test_stories

def count(train_stories,test_stories):#统计所有单词的数量
    vocab = set()
    for story, q, answer in train_stories + test_stories:
        vocab= set(story + q + [answer])
        vocab = sorted(vocab)
    

    # Reserve 0 for masking via pad_sequences
    vocab_size = len(vocab) + 1
    story_maxlen = max(map(len, (x for x, _, _ in train_stories + test_stories)))
    query_maxlen = max(map(len, (x for _, x, _ in train_stories + test_stories)))
    

    print('-')
    print('所有单词的数量:', vocab_size, 'unique words')
    print('所有短文用单词最多的单词数目:', story_maxlen, 'words')
    print('所有问题中最多单词的单词数目:', query_maxlen, 'words')
    print('训练集样本数量:', len(train_stories))
    print('测试集样本数量:', len(test_stories))
    print('-')
    print('Here\'s what a "story" tuple looks like (input, query, answer):')
    print(train_stories[1])
    print('-')
    return vocab,vocab_size,story_maxlen,query_maxlen

def vectorize_stories(data, word_idx, story_maxlen, query_maxlen):
    X = []
    Xq = []
    Y = []
    for story, query, answer in data:
        x = [word_idx[w] for w in story]
        xq = [word_idx[w] for w in query]
        # let's not forget that index 0 is reserved
        y = np.zeros(len(word_idx) + 1)
        y[word_idx[answer]] = 1
        X.append(x)
        Xq.append(xq)
        Y.append(y)
    return (pad_sequences(X, maxlen=story_maxlen),
            pad_sequences(Xq, maxlen=query_maxlen), np.array(Y))

def pad_wordVec(vocab,train_stories,test_stories,story_maxlen,query_maxlen):
    print('Vectorizing the word sequences...')
    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
    inputs_train, queries_train, answers_train = vectorize_stories(train_stories,
                                                               word_idx,
                                                               story_maxlen,
                                                               query_maxlen)
    inputs_test, queries_test, answers_test = vectorize_stories(test_stories,
                                                            word_idx,
                                                            story_maxlen,
                                                            query_maxlen)   
    
    print('-')
    print('inputs: integer tensor of shape (samples, max_length)')
    print('inputs_train shape:', inputs_train.shape)
    print('inputs_test shape:', inputs_test.shape)
    print('-')
    print('queries: integer tensor of shape (samples, max_length)')
    print('queries_train shape:', queries_train.shape)
    print('queries_test shape:', queries_test.shape)
    print('-')
    print('answers: binary (1 or 0) tensor of shape (samples, vocab_size)')
    print('answers_train shape:', answers_train.shape)
    print('answers_test shape:', answers_test.shape)
    print('-')
    print('Compiling...')

if __name__ == '__main__':
 
    path="../../datasets"
    train,test=getData(path)
    print(train[0])
    vocab,vocab_size,story_maxlen,query_maxlen=count(train,test)

    pad_wordVec(vocab,train,test,story_maxlen,query_maxlen)#转化为词向量
   
"""
    #-*- coding:utf8 -*-
    from numpy import *

    #原始数据，训练样本
    def loadDataSet():
        postingList = [
            ['my', 'dog', 'has', 'flea', 'problem', 'help', 'Please'],
            ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
            ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
            ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
            ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
            ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
        ]
        classVec = [0,1,0,1,0,1]
        return postingList, classVec

    #得到所有词的列表
    def createVocabList(dataSet):
        vocabSet = set([])
        for document in dataSet:
            vocabSet = vocabSet | set(document)
        eturn list(vocabSet)

    #某个文档的向量
    def setOfWords2Vec(vocabList, inputSet):
        returnVec = [0]*len(vocabList)
        for word in inputSet:
            if word in vocabList:
                returnVec[vocabList.index(word)] = 1
            else:
                print "the word: %s is not in my Vocabulary!" % word

        return returnVec

#训练函数0
    def trainNB0(trainMatrix, trainCategory):
        numTrainDocs = len(trainMatrix)
        numWords = len(trainMatrix[0])
        pAbusive = sum(trainCategory)/float(numTrainDocs)
        p0Num = zeros(numWords)
        p1Num = zeros(numWords)
        p0Denom = 0.0
        p1Denom = 0.0
        for i in range(numTrainDocs):
            if trainCategory[i] == 1:
                plNum += trainMatrix[i]
                plDenom += sum(trainMatrix)
            else:
                p0Num += trainMatrix[i]
                p0Denom += sum(trainMatrix)
        p0Vect = p0Num/p0Denom
        p1Vect = p1Num/p1Denom
        return p0Vect, p1Vect, pAbusive 
"""

   
    









            
