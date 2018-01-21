# -*- coding: utf-8 -*-

import tensorflow as tf
import sys,os,re
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

from util import NodeLookup
#from VideoCapture import Device

num_top=5

########################################################################  
# #def maybe_download():  
    #print("Downloading Inception v3 Model ...")  
    #download.maybe_download_and_extract(url=data_url, download_dir=data_dir)  
   #如果inception-v3模型不存在就下载，大概85M. 
  
def run_inference_on_image(image_file):
    # 读取图像
    image = tf.gfile.FastGFile(image_file, 'rb').read()
    # 数据层.
    tensor_name_input_jpeg = "DecodeJpeg/contents:0"
    # resize后的数据.
    tensor_name_resized_image = "ResizeBilinear:0"
    # softmax层的名字.
    tensor_name_softmax_logits = "softmax/logits:0"
    # 最后一层的池化.
    tensor_name_transfer_layer = "pool_3:0" 
    # 加载图像分类标签
    #labels = []
    #for label in tf.gfile.GFile("../model/inception-2015-12-05/imagenet_synset_to_human_label_map.txt"):
	    #labels.append(label.rstrip())
    # 加载Graph
    with tf.gfile.FastGFile("../model/inception-2015-12-05/classify_image_graph_def.pb", 'rb') as f:
	    graph_def = tf.GraphDef()
	    graph_def.ParseFromString(f.read())
	    tf.import_graph_def(graph_def, name='')
        # 获取最后softmax层特征数据.
        #self.y_logits = self.graph.get_tensor_by_name(self.tensor_name_softmax_logits)
        # 获取计算图最后一层的数据,可以更改对应名称.
        #self.transfer_layer = self.graph.get_tensor_by_name(self.tensor_name_transfer_layer)
        # 获取最后一层的长度.
        #self.transfer_len = self.transfer_layer.get_shape()[3]
    with tf.Session() as sess:
	    softmax_tensor = sess.graph.get_tensor_by_name('softmax/logits:0')
	    predict = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image})
	    #print(len(predict[0]))
	    #a=np.argmax(predict[0])
	    #print(a)
	    predict=np.squeeze(predict)
        # 根据分类概率进行排序
	    node_lookup = NodeLookup()
	    top = predict.argsort()[-num_top:][::-1]
	    for index in top:
		    human_string=node_lookup.id_to_string(index)
		    score=predict[index]
		    print('%s (score = %.5f)' % (human_string, score))


def main(_):
    # 命令行参数，传入要判断的图片路径
    #image_file = sys.argv[1]
    #cam=Device()
	try:
		while True:
    			#cam.saveSnapshot('')
				run_inference_on_image("./2.jpg")
				print('==================')
	except KeyboardInterrupt:
		pass
if __name__ == '__main__':
    #tf.app.run()


    h = tf.constant("Hello")
    w = tf.constant(" World!")
    hw = h + w

    with tf.Session() as sess:
        ans = sess.run(hw)

    print(ans)
    DATA_DIR = '/tmp/data'
    NUM_STEPS = 1000
    MINIBATCH_SIZE = 100


    data = input_data.read_data_sets(DATA_DIR, one_hot=True)
    X=data.test.images
    Y=data.test.labels
    print(X)
    data = input_data.read_data_sets(DATA_DIR, one_hot=True)

    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))

    y_true = tf.placeholder(tf.float32, [None, 10])
    y_pred = tf.matmul(x, W)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=y_pred, labels=y_true))

    gd_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    correct_mask = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_mask, tf.float32))

    with tf.Session() as sess:

        # Train
        sess.run(tf.global_variables_initializer())
        for _ in range(NUM_STEPS):
            batch_xs, batch_ys = data.train.next_batch(MINIBATCH_SIZE)
            sess.run(gd_step, feed_dict={x: batch_xs, y_true: batch_ys})

        # Test
        ans = sess.run(accuracy, feed_dict={x: data.test.images, y_true: data.test.labels})

    print("Accuracy: {:.4}%".format(ans*100))
		