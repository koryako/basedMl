#from __future__ import division, absolute_import
# -*- coding: utf-8 -*-

#https://www.jianshu.com/p/1909031bb1f2 情感分析
import tflearn
from tflearn.data_utils import to_categorical, pad_sequences, VocabularyProcessor

from sklearn.cross_validation import train_test_split
import pandas as pd
import numpy as np

"""
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
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

# Select only the two columns we require. Game title and its corresponding emotion
dataframe = pd.read_csv('../datasets/loans.csv').ix[:, 1:3]
# Fill null values with empty strings
dataframe.fillna(value='', inplace=True)

def getData_imdb():
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
    return trainX, testX, trainY, testY

def getData(dataframe):
    # Extract the required columns for inputs and outputs
    totalX = dataframe.sub_grade
    totalY = dataframe.short_emp
    print(dataframe.sub_grade.value_counts())
    totalX=totalX.values
    totalY=totalY.values
    vocab_proc = VocabularyProcessor(1)
    # Convert the strings in the input into integers corresponding to the dictionary positions
    # Data is automatically padded so we need to pad_sequences manually
    """
    totalX="I am happy today. I feel sad today."
    vocab_proc = VocabularyProcessor(1)

    totalX = np.array(list(vocab_proc.fit_transform(totalX)))


    #totalX=totalX.transpose()
    #totalX = pad_sequences(totalX, maxlen=100, value=0.)
    print(totalX)
    """
    totalX = np.array(list(vocab_proc.fit_transform(totalX)))

    totalY = to_categorical(totalY, nb_classes=2)#totalY=[1,0,1,0,1] 需要为数组
    # 分离出训练数据和测试数据
    trainX, testX, trainY, testY = train_test_split(totalX, totalY, test_size=0.1)
    return trainX, testX, trainY, testY

    # Network building
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
    #net = tflearn.lstm(net, 128, dropout=0.8)
    # The output is then sent to a fully connected layer that would give us our final 11 classes
    net = tflearn.fully_connected(net, outputlen, activation='softmax')
    # We use the adam optimizer instead of standard SGD since it converges much faster
    net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
						 loss='categorical_crossentropy')
  
    #模型初始化
    model = tflearn.DNN(net, tensorboard_verbose=0)
    return model


trainX, testX, trainY, testY=getData(dataframe)
model=baseline_model(1,2)


def main(train,load_model,save_model):
    # Training
    if train:
        if load_model == 1:
	        model.load('gamemodel.tfl')

        model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True, batch_size=32, n_epoch=20) #show_metric=True #可以看到过程中的准确率
        #result = model.evaluate(test_data, test_labels)
        #print result["accuracy"]

        if save_model == 1:
	        model.save('gamemodel.tfl')
	        print ("Saved model!")
    else:
        model.load('gamemodel.tfl')
        #print ("Predicted %d, Label: %d" % (model.predict(testX[8:9,:]), testY[8:9,:]))
        print (model.predict(testX[8:9,:]), testY[8:9,:])
        #weights = model.weights_ #权重显示
        #a.imshow(weights.T[i].reshape(28, 28), cmap=plt.cm.seismic)

if __name__ == '__main__':
    load_model = 1
    save_model = 1
    train=False
    main(train,load_model,save_model)