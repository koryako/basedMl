# -*- coding: utf-8 -*-

"""

# SqueezeNet Keras Implementation
This is the Keras implementation of SqueezeNet using functional API (arXiv [1602.07360](https://arxiv.org/pdf/1602.07360.pdf)).
SqueezeNet is a small model of AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size.
The original model was implemented in [caffe](https://github.com/DeepScale/SqueezeNet).

## Reference
[pysqueezenet by yhenon](https://github.com/yhenon/pysqueezenet)

Differences:
* Switch from Graph model to Keras 1.0 functional API
* Fix the bug of pooling layer 
* Many thanks to [StefOe](https://github.com/StefOe), the source can now support Keras 2.0 API.

## Result
This repository contains only the Keras implementation of the model, for other parameters used, please see the demo script, [squeezenet_demo.py](https://github.com/tammyyang/simdat/blob/devel/examples/keras/squeezenet_demo.py) in the simdat package.

The training process uses a total of 2,600 images with 1,300 images per class (so, total two classes only).
There are a total 130 images used for validation. After 20 epochs, the model achieves the following:
```
loss: 0.6563 - acc: 0.7065 - val_loss: 0.6247 - val_acc: 0.8750
```
https://github.com/DeepScale/SqueezeNet
https://github.com/DT42/squeezenet_demo
Usage:
1. training
python squeezenet_demo.py --action='train'\
    -p /home/db/train -v /home/db/validation
2. prediction
python squeezenet_demo.py --action='predice'\
    -p /db/Roasted-Broccoli-Pasta-Recipe-5-683x1024.jpg
"""
import time
import json
import argparse
import model as km
#https://github.com/edwardlib/observations
#from simdat.core import dp_tools
from observations import cifar10
from keras.optimizers import Adam
from keras.optimizers import SGD
import numpy as np

#dp = dp_tools.DP()
def generate_batch_data_random(x, y, batch_size):
    """逐步提取batch数据到显存，降低对显存的占用"""
    ylen = len(y)
    loopcount = ylen // batch_size
    while (True):
        i = randint(0,loopcount)
        yield x[i * batch_size:(i + 1) * batch_size], y[i * batch_size:(i + 1) * batch_size]

def generator(arrays, batch_size):
  #Generate batches, one with respect to each array's first axis.
  starts = [0] * len(arrays)
  while True:
    batches = []
    for i, array in enumerate(arrays):
      
      start = starts[i]
      stop = start + batch_size
      diff = stop - array.shape[0]
      if diff <= 0:
        batch = array[start:stop]
        starts[i] += batch_size
      else:
        batch = np.concatenate((array[start:], array[:diff]))
        starts[i] = diff
      batches.append(batch)
    yield batches

"""
model.fit_generator(self.generate_batch_data_random(x_train, y_train, batch_size),                                                      
    samples_per_epoch=len(y_train)//batch_size*batch_size,
    nb_epoch=epoch, 
    validation_data=self.generate_valid_data(x_valid, y_valid,batch_size),
    nb_val_samples=(len(y_valid)//batch_size*batch_size), 
    verbose=verbose,
    callbacks=[early_stopping])

def fit_generator(self, generator, samples_per_epoch, nb_epoch,
                      verbose=1, callbacks=[],
                      validation_data=None, nb_val_samples=None,
                      class_weight=None, max_q_size=10, **kwargs):
""" 
"""
def generator(array, batch_size):
      Generate batch with respect to array's first axis.
  start = 0  # pointer to where we are in iteration
  while True:
    stop = start + batch_size
    diff = stop - array.shape[0]
    if diff <= 0:
      batch = array[start:stop]
      start += batch_size
    else:
      batch = np.concatenate((array[start:], array[:diff]))
      start = diff
    yield batch
"""
"""   
gen_matrix实现从分词后的list来输出训练样本
gen_target实现将输出序列转换为one hot形式的目标
超过maxlen则截断，不足补0
"""
"""
gen_matrix = lambda z: np.vstack((word2vec[z[:maxlen]], np.zeros((maxlen-len(z[:maxlen]), word_size))))
gen_target = lambda z: np_utils.to_categorical(np.array(z[:maxlen] + [0]*(maxlen-len(z[:maxlen]))), 5)

#从节省内存的角度，通过生成器的方式来训练
def data_generator(data, targets, batch_size): 
    idx = np.arange(len(data))
    np.random.shuffle(idx)
    batches = [idx[range(batch_size*i, min(len(data), batch_size*(i+1)))] for i in range(len(data)/batch_size+1)]
    while True:
        for i in batches:
            xx, yy = np.array(map(gen_matrix, data[i])), np.array(map(gen_target, targets[i]))
            yield (xx, yy)


batch_size = 1024
history = model.fit_generator(data_generator(d['words'], d['label'], batch_size), samples_per_epoch=len(d), nb_epoch=200)
model.save_weights('words_seq2seq_final_1.model')
"""
def parse_json(fname):
    """Parse the input profile

    @param fname: input profile path

    @return data: a dictionary with user-defined data for training

    """
    with open(fname) as data_file:
        data = json.load(data_file)
    return data


def oneShot(array,n):
   
    o=[]
    
    for i in range(len(array)):
        p=[0]*n
        p[array[i]-1]=1
        o.append(p)
    return np.array(o)

def write_json(data, fname='./output.json'):
    """Write data to json

    @param data: object to be written

    Keyword arguments:
    fname  -- output filename (default './output.json')

    """
    with open(fname, 'w') as fp:
        json.dump(data, fp, cls=NumpyAwareJSONEncoder)


def print_time(t0, s):
    """Print how much time has been spent

    @param t0: previous timestamp
    @param s: description of this step

    """

    print("%.5f seconds to %s" % ((time.time() - t0), s))
    return time.time()


def main():
    parser = argparse.ArgumentParser(
        description="SqueezeNet example."
        )
    parser.add_argument(
        "--batch-size", type=int, default=32, dest='batchsize',
        help="Size of the mini batch. Default: 32."
        )
    parser.add_argument(
        "--action", type=str, default='train',
        help="Action to be performed, train/predict"
        )
    parser.add_argument(
        "--epochs", type=int, default=20,
        help="Number of epochs, default 20."
        )
    parser.add_argument(
        "--lr", type=float, default=0.001,
        help="Learning rate of SGD, default 0.001."
        )
    parser.add_argument(
        "--epsilon", type=float, default=1e-8,
        help="Epsilon of Adam epsilon, default 1e-8."
        )
    parser.add_argument(
        "-p", "--path", type=str, default='.', required=True,
        help="Path where the images are. Default: $PWD."
        )
    parser.add_argument(
        "-v", "--val-path", type=str, default='.',
        dest='valpath', help="Path where the val images are. Default: $PWD."
        )
    parser.add_argument(
        "--img-width", type=int, default=224, dest='width',
        help="Rows of the images, default: 224."
        )
    parser.add_argument(
        "--img-height", type=int, default=224, dest='height',
        help="Columns of the images, default: 224."
        )
    parser.add_argument(
        "--channels", type=int, default=3,
        help="Channels of the images, default: 3."
        )

    args = parser.parse_args()
    sgd = SGD(lr=args.lr, decay=0.0002, momentum=0.9)
    batch_size=25
    nb_class=10
    t0 = time.time()
    if args.action == 'train':
        (x_train, y_train), (x_test, y_test) = cifar10(args.path)
        y_train=oneShot(y_train,10)
        y_test=oneShot(y_test,10)
        print(y_train[0])
        train_generator = generator([x_train, y_train], batch_size)
        validation_generator=generator([x_test, y_test], batch_size)
        #train_generator = dp.train_data_generator(
            #args.path, args.width, args.height)
        #validation_generator = dp.val_data_generator(
            #args.valpath, args.width, args.height)

        #classes = train_generator.class_indices
        #nb_train_samples = train_generator.samples
        #nb_val_samples = validation_generator.samples
        #print("[squeezenet_demo] N training samples: %i " % nb_train_samples)
        #print("[squeezenet_demo] N validation samples: %i " % nb_val_samples)
        #nb_class = train_generator.num_class
        #print('[squeezenet_demo] Total classes are %i' % nb_class)

        t0 = print_time(t0, 'initialize data')
        model = km.SqueezeNet(
            nb_class, inputs=(args.channels, args.height, args.width))
        # dp.visualize_model(model)
        t0 = print_time(t0, 'build the model')

        model.compile(
            optimizer=sgd, loss='categorical_crossentropy',
            metrics=['accuracy'])
        t0 = print_time(t0, 'compile model')

        model.fit_generator(
            train_generator,
            samples_per_epoch=len(y_train)//batch_size*batch_size,
            nb_epoch=args.epochs,
            validation_data=validation_generator,
            nb_val_samples=(len(y_test)//batch_size*batch_size))
    
        t0 = print_time(t0, 'train model')
        model.save_weights('./weights.h5', overwrite=True)
        """
        model_parms = {'nb_class': nb_class,
                       'nb_train_samples': nb_train_samples,
                       'nb_val_samples': nb_val_samples,
                       'classes': classes,
                       'channels': args.channels,
                       'height': args.height,
                       'width': args.width}
        write_json(model_parms, fname='./model_parms.json')
        """
        t0 = print_time(t0, 'save model')

    elif args.action == 'predict':
        _parms = parse_json('./model_parms.json')
        model = km.SqueezeNet(
            _parms['nb_class'],
            inputs=(_parms['channels'], _parms['height'], _parms['width']),
            weights_path='./weights.h5')
        #dp.visualize_model(model)
        model.compile(
            optimizer=sgd, loss='categorical_crossentropy',
            metrics=['accuracy'])

        X_test, Y_test, classes, F = dp.prepare_data_test(
            args.path, args.width, args.height)
        t0 = print_time(t0, 'prepare data')

        outputs = []
        results = model.predict(
            X_test, batch_size=args.batchsize, verbose=1)
        classes = _parms['classes']
        for i in range(0, len(F)):
            _cls = results[i].argmax()
            max_prob = results[i][_cls]
            outputs.append({'input': F[i], 'max_probability': max_prob})
            cls = [key for key in classes if classes[key] == _cls][0]
            outputs[-1]['class'] = cls
            print('[squeezenet_demo] %s: %s (%.2f)' % (F[i], cls, max_prob))
        t0 = print_time(t0, 'predict')

if __name__ == '__main__':
    main()
