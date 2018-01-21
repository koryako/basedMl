

from __future__ import absolute_import, unicode_literals

import tensorflow as tf
import shutil
import os.path
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("data/", one_hot=True)
# produces the expected result.
x_2 = tf.placeholder("float", shape=[None, 784], name="input")
y__2 = tf.placeholder("float", [None, 10])


with tf.Graph().as_default():
    output_graph_def = tf.GraphDef()
    output_graph_path = './graph/graph.pb'
    #sess.graph.add_to_collection("input", mnist.test.images)
    
    with open(output_graph_path, "rb") as f:
        output_graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(output_graph_def, name="")

    with tf.Session() as sess:
        
        tf.initialize_all_variables().run()
        input_x = sess.graph.get_tensor_by_name("input:0")
        print (input_x)
        output = sess.graph.get_tensor_by_name("output:0")
        print (output)
        
        y_conv_2 = sess.run(output,{input_x:mnist.test.images})
        print ("y_conv_2", y_conv_2)
        
        # Test trained model
        #y__2 = tf.placeholder("float", [None, 10])
        y__2 = mnist.test.labels;
        correct_prediction_2 = tf.equal(tf.argmax(y_conv_2, 1), tf.argmax(y__2, 1))
        print ("correct_prediction_2", correct_prediction_2)
        accuracy_2 = tf.reduce_mean(tf.cast(correct_prediction_2, "float"))
        print ("accuracy_2", accuracy_2)
        
        print ("check accuracy %g" % accuracy_2.eval())