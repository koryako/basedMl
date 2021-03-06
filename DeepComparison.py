# -*- coding: utf-8 -*-

#Load necessary libraries
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
#import input_data
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from matplotlib import cm
import itertools
import numpy as np
from tensorflow.contrib import learn
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import layers
from sklearn import datasets, metrics, preprocessing
from sklearn import model_selection
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Dense, Flatten
from keras.models import Model
from keras.datasets import cifar10
from keras import utils
from keras.datasets import mnist
import sys
from util.minst_input_data import read_data_sets



boston = datasets.load_boston()
x_data = preprocessing.StandardScaler().fit_transform(boston.data)
y_data = boston.target
y_data = y_data.reshape(y_data.shape + (1,))
x_train, x_test, y_train, y_test = model_selection.train_test_split(x_data, y_data, test_size=0.2, random_state=0)

"""

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
    #This function prints and plots the confusion matrix.
    #Normalization can be applied by setting `normalize=True`.
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap,aspect='auto')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


#tf.logging.set_verbosity(tf.logging.ERROR)


DATA_DIR = '../datasets/minist/'

# Load data 

data_one_hot = read_data_sets(DATA_DIR, one_hot=True)

data_classes = read_data_sets(DATA_DIR, one_hot=False)
x_data, y_data = data_classes.train.images,data_classes.train.labels.astype(np.int32)
x_test, y_test = data_classes.test.images,data_classes.test.labels.astype(np.int32)
print(x_data)

print('one-hot=False:\n')
print(data_classes.train.labels)
print('\n')
print('one-hot=True:\n')
print(data_one_hot.train.labels)

feature_columns = [tf.contrib.layers.real_valued_column("", dimension=10)]

dnn = learn.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[200],
    n_classes=10,
    optimizer=tf.train.ProximalAdagradOptimizer(
      learning_rate=0.2,
    ))

fit = dnn.fit(x=x_data,y=y_data, steps=2000,batch_size=128)
print('Done fitting')

ev = dnn.evaluate(x=x_test,y=y_test, steps=1)["accuracy"]
print(ev)

y_pred = dnn.predict(x=x_test,as_iterable=False)
class_names = ['0','1','2','3','4','5','6','7','8','9']    
    
    
# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)
# Plot non-normalized confusion matrix
plt.figure(figsize=[8,6])
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')
"""
"""
learn
from tensorflow.contrib import learn
Estimators
1. Instantiate the Estimator class
model = learn.<Estimator>()
2. Fit it using training data
model.fit()
3. Evaluate how good is the fit
model.evaluate()
4. Predict outcomes on new data
model.predict()
"""

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

currentCifar = 1
cifar = unpickle('../datasets/cifar-10-batches-py/data_batch_1')
cifarT = unpickle('../datasets/cifar-10-batches-py/test_batch')

total_layers = 25 #Specify how deep we want our network
units_between_stride = total_layers / 5


tf.reset_default_graph()

input_layer = tf.placeholder(shape=[None,32,32,3],dtype=tf.float32,name='input')
label_layer = tf.placeholder(shape=[None],dtype=tf.int32)
label_oh = slim.layers.one_hot_encoding(label_layer,10)

layer1 = slim.conv2d(input_layer,64,[3,3],normalizer_fn=slim.batch_norm,scope='conv_'+str(0))
for i in range(5):
    for j in range(units_between_stride):
        layer1 = slim.conv2d(layer1,64,[3,3],normalizer_fn=slim.batch_norm,scope='conv_'+str((j+1) + (i*units_between_stride)))
    layer1 = slim.conv2d(layer1,64,[3,3],stride=[2,2],normalizer_fn=slim.batch_norm,scope='conv_s_'+str(i))
    
top = slim.conv2d(layer1,10,[3,3],normalizer_fn=slim.batch_norm,activation_fn=None,scope='conv_top')

output = slim.layers.softmax(slim.layers.flatten(top))

loss = tf.reduce_mean(-tf.reduce_sum(label_oh * tf.log(output) + 1e-10, axis=[1]))
trainer = tf.train.AdamOptimizer(learning_rate=0.001)
update = trainer.minimize(loss)
"""

def resUnit(input_layer,i):
    with tf.variable_scope("res_unit"+str(i)):
        part1 = slim.batch_norm(input_layer,activation_fn=None)
        part2 = tf.nn.relu(part1)
        part3 = slim.conv2d(part2,64,[3,3],activation_fn=None)
        part4 = slim.batch_norm(part3,activation_fn=None)
        part5 = tf.nn.relu(part4)
        part6 = slim.conv2d(part5,64,[3,3],activation_fn=None)
        output = input_layer + part6
        return output

tf.reset_default_graph()

input_layer = tf.placeholder(shape=[None,32,32,3],dtype=tf.float32,name='input')
label_layer = tf.placeholder(shape=[None],dtype=tf.int32)
label_oh = slim.layers.one_hot_encoding(label_layer,10)

layer1 = slim.conv2d(input_layer,64,[3,3],normalizer_fn=slim.batch_norm,scope='conv_'+str(0))
for i in range(5):
    for j in range(units_between_stride):
        layer1 = resUnit(layer1,j + (i*units_between_stride))
    layer1 = slim.conv2d(layer1,64,[3,3],stride=[2,2],normalizer_fn=slim.batch_norm,scope='conv_s_'+str(i))
    
top = slim.conv2d(layer1,10,[3,3],normalizer_fn=slim.batch_norm,activation_fn=None,scope='conv_top')

output = slim.layers.softmax(slim.layers.flatten(top))

loss = tf.reduce_mean(-tf.reduce_sum(label_oh * tf.log(output) + 1e-10, axis=[1]))
trainer = tf.train.AdamOptimizer(learning_rate=0.001)
update = trainer.minimize(loss)


def highwayUnit(input_layer,i):
    with tf.variable_scope("highway_unit"+str(i)):
        H = slim.conv2d(input_layer,64,[3,3])
        T = slim.conv2d(input_layer,64,[3,3], #We initialize with a negative bias to push the network to use the skip connection
            biases_initializer=tf.constant_initializer(-1.0),activation_fn=tf.nn.sigmoid)
        output = H*T + input_layer*(1.0-T)
        return output

tf.reset_default_graph()

input_layer = tf.placeholder(shape=[None,32,32,3],dtype=tf.float32,name='input')
label_layer = tf.placeholder(shape=[None],dtype=tf.int32)
label_oh = slim.layers.one_hot_encoding(label_layer,10)

layer1 = slim.conv2d(input_layer,64,[3,3],normalizer_fn=slim.batch_norm,scope='conv_'+str(0))
for i in range(5):
    for j in range(units_between_stride):
        layer1 = highwayUnit(layer1,j + (i*units_between_stride))
    layer1 = slim.conv2d(layer1,64,[3,3],stride=[2,2],normalizer_fn=slim.batch_norm,scope='conv_s_'+str(i))
    
top = slim.conv2d(layer1,10,[3,3],normalizer_fn=slim.batch_norm,activation_fn=None,scope='conv_top')

output = slim.layers.softmax(slim.layers.flatten(top))

loss = tf.reduce_mean(-tf.reduce_sum(label_oh * tf.log(output) + 1e-10, axis=[1]))
trainer = tf.train.AdamOptimizer(learning_rate=0.001)
update = trainer.minimize(loss)


def denseBlock(input_layer,i,j):
    with tf.variable_scope("dense_unit"+str(i)):
        nodes = []
        a = slim.conv2d(input_layer,64,[3,3],normalizer_fn=slim.batch_norm)
        nodes.append(a)
        for z in range(j):
            b = slim.conv2d(tf.concat(nodes,3),64,[3,3],normalizer_fn=slim.batch_norm)
            nodes.append(b)
        return b

tf.reset_default_graph()

input_layer = tf.placeholder(shape=[None,32,32,3],dtype=tf.float32,name='input')
label_layer = tf.placeholder(shape=[None],dtype=tf.int32)
label_oh = slim.layers.one_hot_encoding(label_layer,10)

layer1 = slim.conv2d(input_layer,64,[3,3],normalizer_fn=slim.batch_norm,scope='conv_'+str(0))
for i in range(5):
    layer1 = denseBlock(layer1,i,units_between_stride)
    layer1 = slim.conv2d(layer1,64,[3,3],stride=[2,2],normalizer_fn=slim.batch_norm,scope='conv_s_'+str(i))
    
top = slim.conv2d(layer1,10,[3,3],normalizer_fn=slim.batch_norm,activation_fn=None,scope='conv_top')

output = slim.layers.softmax(slim.layers.flatten(top))

loss = tf.reduce_mean(-tf.reduce_sum(label_oh * tf.log(output) + 1e-10, axis=[1]))
trainer = tf.train.AdamOptimizer(learning_rate=0.001)
update = trainer.minimize(loss)


from IPython.display import clear_output, Image, display, HTML

def strip_consts(graph_def, max_const_size=32):
    #Strip large constant values from graph_def.
    strip_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add() 
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = "<stripped %d bytes>"%size
    return strip_def
"""
def show_graph(graph_def, max_const_size=32):
    #Visualize TensorFlow graph.
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    strip_def = strip_consts(graph_def, max_const_size=max_const_size)
    code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:600px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(strip_def)), id='graph'+str(np.random.rand()))

    iframe = """
        <iframe seamless style="width:1200px;height:620px;border:0" srcdoc="{}"></iframe>
    """.format(code.replace('"', '&quot;'))
    display(HTML(iframe))
"""
show_graph(tf.get_default_graph().as_graph_def())


init = tf.global_variables_initializer()


batch_size = 64
currentCifar = 1
total_steps = 20000
l = []
a = []
aT = []
with tf.Session() as sess:
    sess.run(init)
    i = 0
    draw = range(10000)
    while i < total_steps:
        if i % (10000/batch_size) != 0:
            batch_index = np.random.choice(draw,size=batch_size,replace=False)
        else:
            draw = range(10000)
            if currentCifar == 5:
                currentCifar = 1
                print "Switched CIFAR set to " + str(currentCifar)
            else:
                currentCifar = currentCifar + 1
                print "Switched CIFAR set to " + str(currentCifar)
            cifar = unpickle('./cifar10/data_batch_'+str(currentCifar))
            batch_index = np.random.choice(draw,size=batch_size,replace=False)
        x = cifar['data'][batch_index]
        x = np.reshape(x,[batch_size,32,32,3],order='F')
        x = (x/256.0)
        x = (x - np.mean(x,axis=0)) / np.std(x,axis=0)
        y = np.reshape(np.array(cifar['labels'])[batch_index],[batch_size,1])
        _,lossA,yP,LO = sess.run([update,loss,output,label_oh],feed_dict={input_layer:x,label_layer:np.hstack(y)})
        accuracy = np.sum(np.equal(np.hstack(y),np.argmax(yP,1)))/float(len(y))
        l.append(lossA)
        a.append(accuracy)
        if i % 10 == 0: print "Step: " + str(i) + " Loss: " + str(lossA) + " Accuracy: " + str(accuracy)
        if i % 100 == 0: 
            point = np.random.randint(0,10000-500)
            xT = cifarT['data'][point:point+500]
            xT = np.reshape(xT,[500,32,32,3],order='F')
            xT = (xT/256.0)
            xT = (xT - np.mean(xT,axis=0)) / np.std(xT,axis=0)
            yT = np.reshape(np.array(cifarT['labels'])[point:point+500],[500])
            lossT,yP = sess.run([loss,output],feed_dict={input_layer:xT,label_layer:yT})
            accuracy = np.sum(np.equal(yT,np.argmax(yP,1)))/float(len(yT))
            aT.append(accuracy)
            print "Test set accuracy: " + str(accuracy)
        i+= 1

plt.plot(l) #Plot training loss

plt.plot(a) #Plot training accurac
yplt.plot(aT) #Plot test accuracy
np.max(aT) #Best test accuracy

"""