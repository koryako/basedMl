from __future__ import absolute_import
#from __future__ import divison
#from __future__ import print_function

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
flags=tf.app.flags
FLAGS=flags.FLAGS
flags.DEFINE_string('data_dir','data/','Directory for storing data')
mnist=input_data.read_data_sets(FLAGS.data_dir,one_hot=True)

g1=tf.Graph();
vars={}

with g1.as_default():
    with tf.Session() as sess:
         x=tf.placeholder(tf.float32,shape=[None,784],name="input")
         y_=tf.placeholder(tf.float32,shape=[None,10],name="output")
         w=tf.Variable(tf.zeros([784,10]))
         b=tf.Variable(tf.zeros([10]))
         sess.run(tf.initialize_all_variables())
         y=tf.nn.softmax(tf.matmul(x,w)+b)
         cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))
         train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
         for i in range(1000):
             batch = mnist.train.next_batch(50)
             train_step.run(feed_dict={x: batch[0], y_: batch[1]})
             for v in tf.trainable_variables():
                 vars[v.value().name] = sess.run(v)
             correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
             accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
             print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
g2 = tf.Graph()
consts = {}
with g2.as_default():
    with tf.Session() as sess:
         for k in vars.keys():
               consts[k] = tf.constant(vars[k])
         tf.import_graph_def(g1.as_graph_def(),input_map={name:consts[name] for name in consts.keys()})
         tf.train.write_graph(sess.graph_def,'graph/','graph.pb',False)