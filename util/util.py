# -*- coding: utf-8 -*-
import os,re
import numpy as np
import tensorflow as tf
def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

def load_cifar_10(rootdir):
     offset=10000
     X=np.zeros((50000,3072))
     list = os.listdir(rootdir) 
     for i in range(0,len(list)):
        path = os.path.join(rootdir,'data_batch_{}'.format(i))
        if os.path.isfile(path):
            d=unpickle(path)
            n=i-1
            X[n*offset:(n+1)*offset,:]=d['data']
            Y=np.array(d['labels'])
     return X,Y

def load_cvs_manuel_earthquake():
    data_total = []
    with open('../datasets/database.csv', "r") as f:
        f.readline()
        for line in f:
            data_point = []
            values = line.split(',')
            cur_date = values[0].split('/')
            # Consider only the year and month since they are highly significant
            # Time can be ignored.
            if len(cur_date) < 3:
                continue
            cur_year = cur_date[2]
            cur_month = cur_date[1]
            data_point.append(float(cur_year))#获取年
            data_point.append(float(cur_month))#获取日
            # Latitude
            data_point.append(float(values[2]))
            # Longitude
            data_point.append(float(values[3]))
            # Magnitude
            data_point.append(float(values[8]))
            data_total.append(data_point)
    return np.asarray(data_total)

# Normalize the data

def Normalize(data_total):
   means = data_total.mean(axis=0) #列数据平均数
   variance = data_total.std(axis=0)#列数据标准值
   data_total -= means
   data_total /= variance
   return data_total

def val_test(data_total,rate):
    p=10
    v=int(10*rate)
    t=data_total.shape[1]-1
    # Split the data into 80%, 20% training and testing sets
    data_train = data_total[:(len(data_total)*v)/p]
    data_test = data_total[(len(data_total)*v)/p:len(data_total)-1]

    data_train_x, data_train_y = data_train[:,:t], data_train[:,t]
    data_test_x, data_test_y = data_test[:,:t], data_test[:,t]
   
    data_train_y = data_train_y.reshape((len(data_train_y), 1))
    data_test_y = data_test_y.reshape((len(data_test_y), 1))
    return data_train_x,data_train_y,data_test_x,data_test_y


model_dir='../model/inception-2015-12-05'

#Converts integer node ID's to human readable labels.
class NodeLookup(object):
    
    def __init__(self,label_lookup_path=None,uid_lookup_path=None):
        if label_lookup_path is None:
            label_lookup_path = os.path.join(model_dir, 'imagenet_2012_challenge_label_map_proto.pbtxt')
        if not uid_lookup_path:
            uid_lookup_path = os.path.join(model_dir, 'imagenet_synset_to_human_label_map.txt')
        self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

    def load(self, label_lookup_path, uid_lookup_path):
        if not tf.gfile.Exists(uid_lookup_path):
           tf.logging.fatal('File does not exist %s', uid_lookup_path)
        if not tf.gfile.Exists(label_lookup_path):
           tf.logging.fatal('File does not exist %s', label_lookup_path)

        # Loads mapping from string UID to human-readable string
        proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
        uid_to_human = {}
        p = re.compile(r'[n\d]*[ \S,]*')
        for line in proto_as_ascii_lines:
           parsed_items = p.findall(line)
           uid = parsed_items[0]
           human_string = parsed_items[2]
           uid_to_human[uid] = human_string

    # Loads mapping from string UID to integer node ID.
        node_id_to_uid = {}
        proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
        for line in proto_as_ascii:
           if line.startswith('  target_class:'):
              target_class = int(line.split(': ')[1])
           if line.startswith('  target_class_string:'):
              target_class_string = line.split(': ')[1]
              node_id_to_uid[target_class] = target_class_string[1:-2]

    # Loads the final mapping of integer node ID to human-readable string
        node_id_to_name = {}
        for key, val in node_id_to_uid.items():
            if val not in uid_to_human:
               tf.logging.fatal('Failed to locate: %s', val)
            name = uid_to_human[val]
            node_id_to_name[key] = name

        return node_id_to_name

    def id_to_string(self, node_id):
        if node_id not in self.node_lookup:
           return ''
        return self.node_lookup[node_id]

if __name__ == '__main__':
    data,label=load_cifar_10('../datasets/cifar-10-batches-py')
    print(data.shape)
    print(label.shape)