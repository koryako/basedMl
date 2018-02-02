#from __future__ import division, absolute_import
# -*- coding: utf-8 -*-

#https://www.jianshu.com/p/1909031bb1f2 情感分析
import tflearn
from tflearn.data_utils import to_categorical, pad_sequences, VocabularyProcessor

from sklearn.cross_validation import train_test_split
import pandas as pd
import numpy as np

load_model = 1
save_model = 1


# Select only the two columns we require. Game title and its corresponding emotion
dataframe = pd.read_csv('../datasets/loans.csv').ix[:, 1:3]
# Fill null values with empty strings
dataframe.fillna(value='', inplace=True)

print(dataframe.sub_grade.value_counts())

# Extract the required columns for inputs and outputs
totalX = dataframe.sub_grade
totalY = dataframe.short_emp
print(totalX.values)
print(totalY.values)
totalX=totalX.values
totalY=totalY.values
vocab_proc = VocabularyProcessor(1)

totalX = np.array(list(vocab_proc.fit_transform(totalX)))
#print(totalX)
totalY = to_categorical(totalY, nb_classes=2)
#print(totalY)
# Convert the strings in the input into integers corresponding to the dictionary positions
# Data is automatically padded so we need to pad_sequences manually



# Split into training and testing data
trainX, testX, trainY, testY = train_test_split(totalX, totalY, test_size=0.1)

print(trainX)
print(trainY)
print(testX[0,:])

def baseline_model(inputlen,outputlen):#第二个参数为输出分类数目
    # Build the network for classification
    # Each input has length of 15
    net = tflearn.input_data([None, inputlen])#每个样本整数向量的长度，也就是每行单词的排序索引 长度是每句话的最大单词数
    # The 15 input word integers are then casted out into 256 dimensions each creating a word embedding.
    # We assume the dictionary has 10000 words maximum
    net = tflearn.embedding(net, input_dim=10000, output_dim=256)
    # Each input would have a size of 15x256 and each of these 256 sized vectors are fed into the LSTM layer one at a time.
    # All the intermediate outputs are collected and then passed on to the second LSTM layer.
    net = tflearn.gru(net, 256, dropout=0.9, return_seq=True)
    # Using the intermediate outputs, we pass them to another LSTM layer and collect the final output only this time
    net = tflearn.gru(net, 256, dropout=0.9)
    # The output is then sent to a fully connected layer that would give us our final 11 classes
    net = tflearn.fully_connected(net, outputlen, activation='softmax')
    # We use the adam optimizer instead of standard SGD since it converges much faster
    net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
						 loss='categorical_crossentropy')
    # Train the network
    model = tflearn.DNN(net, tensorboard_verbose=0)
    return model


model=baseline_model(1,2)
train=False

if train:
    if load_model == 1:
	    model.load('gamemodel.tfl')

    model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True, batch_size=32, n_epoch=1)

    if save_model == 1:
	    model.save('gamemodel.tfl')
	    print ("Saved model!")
else:
    model.load('gamemodel.tfl')
    print (model.predict(testX[8:9,:]), testY[8:9,:])
"""
result = classifier.evaluate(test_data, test_labels)
print result["accuracy"]


# here's one it gets right

display(0)
# and one it gets wrong
print ("Predicted %d, Label: %d" % (classifier.predict(test_data[8]), test_labels[8]))
display(8)

weights = classifier.weights_
a.imshow(weights.T[i].reshape(28, 28), cmap=plt.cm.seismic)
"""

"""
text="I am happy today. I feel sad today."
from textblob import TextBlob
blob=TextBlob(text)
blob.sentences
blob.sentences[0].sentiment

blob.sentences[1].sentiment

blob.sentiment

chinese=u"我今天很快乐。 我今天很愤怒。"
from snownlp import SnowNLP
s=SnowNLP(chinese)
for sentence in s.sentences:
    print(sentence)

s1=SnowNLP(s.sentences[1])
s1.sentiments
"""
"""
from tflearn.datasets import imdb

train, test, _ = imdb.load_data(path='imdb.pkl', n_words=10000,
                                valid_portion=0.1)
trainX, trainY = train
testX, testY = test
# Data preprocessing
# Sequence padding
trainX = pad_sequences(trainX, maxlen=100, value=0.)
print(trainX.shape)
print(trainY)
testX = pad_sequences(testX, maxlen=100, value=0.)
# Converting labels to binary vectors
trainY = to_categorical(trainY, nb_classes=2)
testY = to_categorical(testY, nb_classes=2)
# Network building
net = tflearn.input_data([None, 100])

net = tflearn.embedding(net, input_dim=10000, output_dim=128)
net = tflearn.lstm(net, 128, dropout=0.8)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                         loss='categorical_crossentropy')

# Training
#模型初始化
model = tflearn.DNN(net, tensorboard_verbose=0)

show_metric=True #可以看到过程中的准确率
model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True,
          batch_size=32)
"""
"""
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
"""
"""
totalX="I am happy today. I feel sad today."
vocab_proc = VocabularyProcessor(1)

totalX = np.array(list(vocab_proc.fit_transform(totalX)))


totalX=totalX.transpose()
totalX = pad_sequences(totalX, maxlen=100, value=0.)
print(totalX)
totalY=[1,0,1,0,1]
totalY = to_categorical(totalY, nb_classes=2)
print(totalY)
# We will have 11 classes in total for prediction, indices from 0 to 10
#vocab_proc2 = VocabularyProcessor(1)
#totalY = np.array(list(vocab_proc2.fit_transform(totalY))) - 1
# Convert the indices into 11 dimensional vectors
#totalY = to_categorical(totalY, nb_classes=11)
"""