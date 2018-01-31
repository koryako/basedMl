#opencv3.3
#https://www.pyimagesearch.com/2017/09/11/object-detection-with-deep-learning-and-opencv/
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse as arg
import imutils
import time
import cv2

ap=arg.ArgumentParser()
ap.add_argument("-p","--prototxt",required=True,help="caffe'deploy'prototxt file")
ap.add_argument('-m',"-model",required=True,help="Caffe pre-trained model")
ap.add_argument('-c',"--confidence",type=float,default=0.2,help="过滤弱检测的最小概率")
args=vars(ap.parse_args())
CLASSES=['background','horse','horse','person','bicycle']
COLARS=np.random.uniform(0,255,size=(len(CLASSES),3))

print('[INFO] loading model...')
net=cv2.dnn.readNetFromCaffe(args["prototext"],ars['model'])
print ('[INFO] starting video stream')

vs=VideoStream(src=1).start()
time.sleep(2.0)
fps=FPS().start()

def draw():
    for i in np.arange(0,detections.shae[2]):
        confidence=detections[0,0,i,2]
        if confidence>args["confidence"]:
            idx=int(detections[0,0,i,1])
            box=detections[0,0,i,3:7]*np.array([w,h,w,h])
            (startX,startY,endX,endY)=box.astype('int')

while True:
    frame=vs.read()
    frame=imutils.resize(frame,width=400)
    (h,w)=frame.shape[:2]
    blob=cv2.dnn.blobFromImage(frame,0.007843,(300,300),127.5)
    net.setInput(blob)
    detections=net.forward()

"""
tensorflow 对mobilenet重新训练

python tensorflow/examples/image_retraining/retrain.py \
    --image_dir ~/ml/blogs/road-not-road/data/ \
    --learning_rate=0.0001 \
    --testing_percentage=20 \
    --validation_percentage=20 \
    --train_batch_size=32 \
    --validation_batch_size=-1 \
    --flip_left_right True \
    --random_scale=30 \
    --random_brightness=30 \
    --eval_step_interval=100 \
    --how_many_training_steps=600 \
    --architecture mobilenet_1.0_224

https://github.com/marc-antimodular/ofxOpenCV-MobileNetSDD-Example

https://github.com/opencv/opencv/blob/master/samples/dnn/ssd_mobilenet_object_detection.cpp

http://blog.csdn.net/wfei101/article/details/78310226 mobile理解
Tensoflow mobilenet 分类

http://blog.csdn.net/u010302327/article/details/78248394 下载voc 数据集

https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md 预训练模型

http://www.image-net.org/challenges/LSVRC/2012/  imagenet


https://github.com/Zehaos/MobileNet 非官方mobilenet

https://github.com/chuanqi305/MobileNet-SSd  caffe版本 有预训练模型


https://github.com/weiliu89/caffe/tree/ssd 使用非MobelNet 训练的ssd

https://github.com/JianGoForIt/YellowFin  momentum SGD 优化器
https://github.com/matvi/ObjectDetectionMobileNetSSD caffe 版本
http://blog.csdn.net/xxiaozr/article/details/77073164 ssd理解

https://github.com/phongnhhn92/Remote-Marker-based-Tracking-using-MobileNet-SSD 物联网应用

https://developforai.com/ 案例
Densnet
-------------
Densely Connected Convolutional Networks》当选 CVPR 2017 最佳论文，
Torch implementation: https://github.com/liuzhuang13/DenseNet/tree/master/models

PyTorch implementation: https://github.com/gpleiss/efficient_densenet_pytorch

MxNet implementation: https://github.com/taineleau/efficient_densenet_mxnet

Caffe implementation: https://github.com/Tongcheng/DN_CaffeScript
与苹果的首篇公开论文《Learning From Simulated and Unsupervised Images through Adversarial Training》共获这一殊荣。
CVPR 2017 的一篇 workshop 文章 《The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation》 (https://arxiv.org/abs/1611.09326) 表明，基于 DenseNet 的全卷积网络（FCN）模型在不需要预训练的情况下甚至可以达到比其他预训练方法更高的精度，并且比达到相同效果的其他方法的模型要小 10 倍。

https://github.com/Queequeg92/SE-Net-CIFAR  se-net 代码

https://github.com/szq0214/DSOD  Densnet+ssd

https://github.com/songrotek/Deep-Learning-Papers-Reading-Roadmap 机器学习各领域早期论文
-------------------
https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/android/
https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/inception5h.py

https://machinelearningflashcards.com/ tick
https://chrisalbon.com/
https://www.kdnuggets.com/2017/09/neural-networks-tic-tac-toe-keras.html
keras例子
https://chrisalbon.com/ 博客
https://github.com/keras-team/keras/tree/master/examples

https://www.kdnuggets.com/tutorials/index.html 英文学习资料
"""
#这里的网络架构和论文中插图中的网络架构是相一致的。
#对了，忘了说了，这里使用的keras版本是1.2.2，等源码读完之后，我自己改一个2.0.6版本上传到github上面。
#可别直接粘贴复制，里面有些中文的解释，不一定可行的。
#defint input shape
input_shape = (300,300,3)
#defint the number of classes
num_classes = 21

#Here the network is wrapped in to a dictory because it more easy to make some operations.
net = {}
# Block 1
input_tensor = Input(shape=input_shape)
#defint the image hight and wight
img_size = (input_shape[1], input_shape[0])
net['input'] = input_tensor
net['conv1_1'] = Convolution2D(64, 3, 3,
                               activation='relu',
                               border_mode='same',
                               name='conv1_1')(net['input'])
net['conv1_2'] = Convolution2D(64, 3, 3,
                               activation='relu',
                               border_mode='same',
                               name='conv1_2')(net['conv1_1'])
net['pool1'] = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same',
                            name='pool1')(net['conv1_2'])
# Block 2
net['conv2_1'] = Convolution2D(128, 3, 3,
                               activation='relu',
                               border_mode='same',
                               name='conv2_1')(net['pool1'])
net['conv2_2'] = Convolution2D(128, 3, 3,
                               activation='relu',
                               border_mode='same',
                               name='conv2_2')(net['conv2_1'])
net['pool2'] = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same',
                            name='pool2')(net['conv2_2'])
# Block 3
net['conv3_1'] = Convolution2D(256, 3, 3,
                               activation='relu',
                               border_mode='same',
                               name='conv3_1')(net['pool2'])
net['conv3_2'] = Convolution2D(256, 3, 3,
                               activation='relu',
                               border_mode='same',
                               name='conv3_2')(net['conv3_1'])
net['conv3_3'] = Convolution2D(256, 3, 3,
                               activation='relu',
                               border_mode='same',
                               name='conv3_3')(net['conv3_2'])
net['pool3'] = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same',
                            name='pool3')(net['conv3_3'])
# Block 4
net['conv4_1'] = Convolution2D(512, 3, 3,
                               activation='relu',
                               border_mode='same',
                               name='conv4_1')(net['pool3'])
net['conv4_2'] = Convolution2D(512, 3, 3,
                               activation='relu',
                               border_mode='same',
                               name='conv4_2')(net['conv4_1'])
#the first layer be operated
net['conv4_3'] = Convolution2D(512, 3, 3,
                               activation='relu',
                               border_mode='same',
                               name='conv4_3')(net['conv4_2'])
net['pool4'] = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same',
                            name='pool4')(net['conv4_3'])
# Block 5
net['conv5_1'] = Convolution2D(512, 3, 3,
                               activation='relu',
                               border_mode='same',
                               name='conv5_1')(net['pool4'])
net['conv5_2'] = Convolution2D(512, 3, 3,
                               activation='relu',
                               border_mode='same',
                               name='conv5_2')(net['conv5_1'])
net['conv5_3'] = Convolution2D(512, 3, 3,
                               activation='relu',
                               border_mode='same',
                               name='conv5_3')(net['conv5_2'])
net['pool5'] = MaxPooling2D((3, 3), strides=(1, 1), border_mode='same',
                            name='pool5')(net['conv5_3'])
#here is the FC6 in the orginal VGG16 Network，There move to Atrous Convolution for the reason i don't know.
# FC6
net['fc6'] = AtrousConvolution2D(1024, 3, 3, atrous_rate=(6, 6),
                                 activation='relu', border_mode='same',
                                 name='fc6')(net['pool5'])
#the second layer to be operated
# FC7
net['fc7'] = Convolution2D(1024, 1, 1, activation='relu',
                           border_mode='same', name='fc7')(net['fc6'])
# x = Dropout(0.5, name='drop7')(x)
# Block 6
net['conv6_1'] = Convolution2D(256, 1, 1, activation='relu',
                               border_mode='same',
                               name='conv6_1')(net['fc7'])
#the third layer to be opreated
net['conv6_2'] = Convolution2D(512, 3, 3, subsample=(2, 2),
                               activation='relu', border_mode='same',
                               name='conv6_2')(net['conv6_1'])
# Block 7
net['conv7_1'] = Convolution2D(128, 1, 1, activation='relu',
                               border_mode='same',
                               name='conv7_1')(net['conv6_2'])
net['conv7_2'] = ZeroPadding2D()(net['conv7_1'])
#the forth layer to be operated
net['conv7_2'] = Convolution2D(256, 3, 3, subsample=(2, 2),
                               activation='relu', border_mode='valid',
                               name='conv7_2')(net['conv7_2'])
# Block 8
net['conv8_1'] = Convolution2D(128, 1, 1, activation='relu',
                               border_mode='same',
                               name='conv8_1')(net['conv7_2'])
#the fifth layer to be operated
net['conv8_2'] = Convolution2D(256, 3, 3, subsample=(2, 2),
                               activation='relu', border_mode='same',
                               name='conv8_2')(net['conv8_1'])
# the last layer to be operated
# Last Pool 
net['pool6'] = GlobalAveragePooling2D(name='pool6')(net['conv8_2'])

# Prediction from conv4_3
# net['conv4_3']._shape = (?, 38, 38, 512)
# 算了还是说中文吧，这个层是用来对输入数据进行正则化的层，有参数需要学习，输出的数据形式和输入输入形式是一致的。
net['conv4_3_norm'] = Normalize(20, name='conv4_3_norm')(net['conv4_3'])
num_priors = 3
#here is *4 because the box need 4 number to define，here is only predice the box coordinate
x = Convolution2D(num_priors * 4, 3, 3, border_mode='same',
                  name='conv4_3_norm_mbox_loc')(net['conv4_3_norm'])
net['conv4_3_norm_mbox_loc'] = x
flatten = Flatten(name='conv4_3_norm_mbox_loc_flat')
net['conv4_3_norm_mbox_loc_flat'] = flatten(net['conv4_3_norm_mbox_loc'])
#the box coordinate is finished now it will perdice the classes
name = 'conv4_3_norm_mbox_conf'
if num_classes != 21:
    name += '_{}'.format(num_classes)
# here is start predict the classes
x = Convolution2D(num_priors * num_classes, 3, 3, border_mode='same',
                  name=name)(net['conv4_3_norm'])
net['conv4_3_norm_mbox_conf'] = x
flatten = Flatten(name='conv4_3_norm_mbox_conf_flat')
net['conv4_3_norm_mbox_conf_flat'] = flatten(net['conv4_3_norm_mbox_conf'])
#这里是用来对conv4_3层的feature map生成论文中所说的default box，对没错，就是直接使用Feature map来进行default box的生成
#当然这里要指定一些参数，这些参数是需要好好斟酌的。
priorbox = PriorBox(img_size, 30.0, aspect_ratios=[2],
                    variances=[0.1, 0.1, 0.2, 0.2],
                    name='conv4_3_norm_mbox_priorbox')
net['conv4_3_norm_mbox_priorbox'] = priorbox(net['conv4_3_norm'])
#好了，到这里第一个层的操作就完成了，下面其他层的操作都是相类似的啦。
# Prediction from fc7
num_priors = 6
net['fc7_mbox_loc'] = Convolution2D(num_priors * 4, 3, 3,
                                    border_mode='same',
                                    name='fc7_mbox_loc')(net['fc7'])
flatten = Flatten(name='fc7_mbox_loc_flat')
net['fc7_mbox_loc_flat'] = flatten(net['fc7_mbox_loc'])
name = 'fc7_mbox_conf'
if num_classes != 21:
    name += '_{}'.format(num_classes)
net['fc7_mbox_conf'] = Convolution2D(num_priors * num_classes, 3, 3,
                                     border_mode='same',
                                     name=name)(net['fc7'])
flatten = Flatten(name='fc7_mbox_conf_flat')
net['fc7_mbox_conf_flat'] = flatten(net['fc7_mbox_conf'])
priorbox = PriorBox(img_size, 60.0, max_size=114.0, aspect_ratios=[2, 3],
                    variances=[0.1, 0.1, 0.2, 0.2],
                    name='fc7_mbox_priorbox')
net['fc7_mbox_priorbox'] = priorbox(net['fc7'])
# Prediction from conv6_2
num_priors = 6
x = Convolution2D(num_priors * 4, 3, 3, border_mode='same',
                  name='conv6_2_mbox_loc')(net['conv6_2'])
net['conv6_2_mbox_loc'] = x
flatten = Flatten(name='conv6_2_mbox_loc_flat')
net['conv6_2_mbox_loc_flat'] = flatten(net['conv6_2_mbox_loc'])
name = 'conv6_2_mbox_conf'
if num_classes != 21:
    name += '_{}'.format(num_classes)
x = Convolution2D(num_priors * num_classes, 3, 3, border_mode='same',
                  name=name)(net['conv6_2'])
net['conv6_2_mbox_conf'] = x
flatten = Flatten(name='conv6_2_mbox_conf_flat')
net['conv6_2_mbox_conf_flat'] = flatten(net['conv6_2_mbox_conf'])
priorbox = PriorBox(img_size, 114.0, max_size=168.0, aspect_ratios=[2, 3],
                    variances=[0.1, 0.1, 0.2, 0.2],
                    name='conv6_2_mbox_priorbox')
net['conv6_2_mbox_priorbox'] = priorbox(net['conv6_2'])
# Prediction from conv7_2
num_priors = 6
x = Convolution2D(num_priors * 4, 3, 3, border_mode='same',
                  name='conv7_2_mbox_loc')(net['conv7_2'])
net['conv7_2_mbox_loc'] = x
flatten = Flatten(name='conv7_2_mbox_loc_flat')
net['conv7_2_mbox_loc_flat'] = flatten(net['conv7_2_mbox_loc'])
name = 'conv7_2_mbox_conf'
if num_classes != 21:
    name += '_{}'.format(num_classes)
x = Convolution2D(num_priors * num_classes, 3, 3, border_mode='same',
                  name=name)(net['conv7_2'])
net['conv7_2_mbox_conf'] = x
flatten = Flatten(name='conv7_2_mbox_conf_flat')
net['conv7_2_mbox_conf_flat'] = flatten(net['conv7_2_mbox_conf'])
priorbox = PriorBox(img_size, 168.0, max_size=222.0, aspect_ratios=[2, 3],
                    variances=[0.1, 0.1, 0.2, 0.2],
                    name='conv7_2_mbox_priorbox')
net['conv7_2_mbox_priorbox'] = priorbox(net['conv7_2'])
# Prediction from conv8_2
num_priors = 6
x = Convolution2D(num_priors * 4, 3, 3, border_mode='same',
                  name='conv8_2_mbox_loc')(net['conv8_2'])
net['conv8_2_mbox_loc'] = x
flatten = Flatten(name='conv8_2_mbox_loc_flat')
net['conv8_2_mbox_loc_flat'] = flatten(net['conv8_2_mbox_loc'])
name = 'conv8_2_mbox_conf'
if num_classes != 21:
    name += '_{}'.format(num_classes)
x = Convolution2D(num_priors * num_classes, 3, 3, border_mode='same',
                  name=name)(net['conv8_2'])
net['conv8_2_mbox_conf'] = x
flatten = Flatten(name='conv8_2_mbox_conf_flat')
net['conv8_2_mbox_conf_flat'] = flatten(net['conv8_2_mbox_conf'])
priorbox = PriorBox(img_size, 222.0, max_size=276.0, aspect_ratios=[2, 3],
                    variances=[0.1, 0.1, 0.2, 0.2],
                    name='conv8_2_mbox_priorbox')
net['conv8_2_mbox_priorbox'] = priorbox(net['conv8_2'])
# Prediction from pool6
num_priors = 6
x = Dense(num_priors * 4, name='pool6_mbox_loc_flat')(net['pool6'])
net['pool6_mbox_loc_flat'] = x
name = 'pool6_mbox_conf_flat'
if num_classes != 21:
    name += '_{}'.format(num_classes)
x = Dense(num_priors * num_classes, name=name)(net['pool6'])
net['pool6_mbox_conf_flat'] = x
priorbox = PriorBox(img_size, 276.0, max_size=330.0, aspect_ratios=[2, 3],
                    variances=[0.1, 0.1, 0.2, 0.2],
                    name='pool6_mbox_priorbox')
#由于这里的维数不对，因此要修改Feature map层对应的维数信息
if K.image_dim_ordering() == 'tf':
    target_shape = (1, 1, 256)
else:
    target_shape = (256, 1, 1)
net['pool6_reshaped'] = Reshape(target_shape,
                                name='pool6_reshaped')(net['pool6'])
net['pool6_mbox_priorbox'] = priorbox(net['pool6_reshaped'])
#好啦，到这里位置，所有的信息都已经生成了，下一步就是根据这些信息来进行训练或者是预测了。
# Gather all predictions
net['mbox_loc'] = merge([net['conv4_3_norm_mbox_loc_flat'],
                         net['fc7_mbox_loc_flat'],
                         net['conv6_2_mbox_loc_flat'],
                         net['conv7_2_mbox_loc_flat'],
                         net['conv8_2_mbox_loc_flat'],
                         net['pool6_mbox_loc_flat']],
                        mode='concat', concat_axis=1, name='mbox_loc')
net['mbox_conf'] = merge([net['conv4_3_norm_mbox_conf_flat'],
                          net['fc7_mbox_conf_flat'],
                          net['conv6_2_mbox_conf_flat'],
                          net['conv7_2_mbox_conf_flat'],
                          net['conv8_2_mbox_conf_flat'],
                          net['pool6_mbox_conf_flat']],
                         mode='concat', concat_axis=1, name='mbox_conf')
net['mbox_priorbox'] = merge([net['conv4_3_norm_mbox_priorbox'],
                              net['fc7_mbox_priorbox'],
                              net['conv6_2_mbox_priorbox'],
                              net['conv7_2_mbox_priorbox'],
                              net['conv8_2_mbox_priorbox'],
                              net['pool6_mbox_priorbox']],
                             mode='concat', concat_axis=1,
                             name='mbox_priorbox')
if hasattr(net['mbox_loc'], '_keras_shape'):
    num_boxes = net['mbox_loc']._keras_shape[-1] // 4
elif hasattr(net['mbox_loc'], 'int_shape'):
    num_boxes = K.int_shape(net['mbox_loc'])[-1] // 4
net['mbox_loc'] = Reshape((num_boxes, 4),
                          name='mbox_loc_final')(net['mbox_loc'])
net['mbox_conf'] = Reshape((num_boxes, num_classes),
                           name='mbox_conf_logits')(net['mbox_conf'])
net['mbox_conf'] = Activation('softmax',
                              name='mbox_conf_final')(net['mbox_conf'])
net['predictions'] = merge([net['mbox_loc'],
                           net['mbox_conf'],
                           net['mbox_priorbox']],
                           mode='concat', concat_axis=2,
                           name='predictions')
model = Model(net['input'], net['predictions'])





