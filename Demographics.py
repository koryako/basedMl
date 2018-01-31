# -*- coding: utf-8 -*-
# 超市购买用户分类预测（购买时间序列）
import keras
from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam

import tensorflow as tf

from sklearn.preprocessing import LabelEncoder

import numpy as np
import pandas as pd 

def make_sequence(num_users, num_categories, max_len):
    category_sequences = [
        np.random.randint(0, num_categories, np.random.randint(max_len)) 
        for _ in range(num_users)
    ]
    return category_sequences
num_users = 1000
max_len = 50
num_categories = 100
category_sequences = make_sequence(num_users, num_categories, max_len)#随机取0-100 种类别， 每次类别种类也不一样，向量长度不一样，一共1000行数据
print(category_sequences[0])

def to_fractions(sequence):
    def row_to_fractions(row):
        return pd.Series(row).value_counts() / len(row)
    
    return pd.DataFrame([row_to_fractions(seq) for seq in sequence]).fillna(0).values# 一行序列 每个类别出现的概率


frac = to_fractions(category_sequences)
print(frac[0])

#keras.layers.Embedding(num_categories, embedding_dim)

#embedding_matrix = tf.get_variable("embeddings", [num_categories, embedding_dim])
#embeddings = tf.nn.embedding_lookup(embedding_matrix, category_sequence_goes_here)


def deep_user_multiple_sequences(input_sizes, output_sizes, embedding_sizes, depth=(100, 100)):
    
    # The inputs are not actually sequences! they are the distribution over sequence objects...
    inputs = [Input(shape=(s,)) for s in input_sizes]

    # Each input is then embedded into its own space 
    # (relu not really necessary...)
    embeddings = [Dense(emb_size, activation='relu')(input) 
                  for emb_size, input in zip(embedding_sizes, inputs)]

    # Concat everything
    everything = concatenate(embeddings)

    # Add in additional layers
    for layer_size in depth:
        everything = Dense(layer_size, activation='relu')(everything)

    # Go to output
    outputs = [Dense(out_size, activation='softmax')(everything) 
               for out_size in output_sizes]

    # Build, print, and return model
    model = Model(inputs=inputs, outputs=outputs)
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return model


my_model = deep_user_multiple_sequences(input_sizes=(100, 50), 
                                        output_sizes=(4, 3), 
                                        embedding_sizes=(100, 100))


y_marital = np.random.choice(["single", "married", "divorced", "widowed"], num_users)
y_children = np.random.choice(["1", "2", "2+"], num_users)

y_marital = np.eye(4)[LabelEncoder().fit_transform(y_marital)]
y_children = np.eye(3)[LabelEncoder().fit_transform(y_children)]
print(y_marital)
print(y_children)

seq1 = to_fractions(make_sequence(num_users, num_categories=100, max_len=500))
seq2 = to_fractions(make_sequence(num_users, num_categories=50, max_len=500))

my_model.fit([seq1, seq2], [y_marital, y_children], epochs=200)