from keras.models import Sequentialfrom scipy.misc import imread
get_ipython().magic('matplotlib inline')import matplotlib.pyplot as pltimport numpy as npimport kerasfrom keras.layers import Denseimport pandas as pdfrom keras.applications.vgg16 import VGG16from keras.preprocessing import imagefrom keras.applications.vgg16 import preprocess_inputimport numpy as npfrom keras.applications.vgg16 import decode_predictionsfrom keras.utils.np_utils import to_categoricalfrom sklearn.preprocessing import LabelEncoderfrom keras.models import Sequentialfrom keras.optimizers import SGDfrom keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activationfrom sklearn.metrics import log_loss

train=pd.read_csv("R/Data/Train/train.csv")
test=pd.read_csv("R/Data/test.csv")
train_path="R/Data/Train/Images/train/"test_path="R/Data/Train/Images/test/"from scipy.misc import imresize

train_img=[]for i in range(len(train)):

    temp_img=image.load_img(train_path+train['filename'][i],target_size=(224,224))

    temp_img=image.img_to_array(temp_img)

    train_img.append(temp_img)

train_img=np.array(train_img) 
train_img=preprocess_input(train_img)

test_img=[]for i in range(len(test)):

temp_img=image.load_img(test_path+test['filename'][i],target_size=(224,224))

    temp_img=image.img_to_array(temp_img)

    test_img.append(temp_img)

test_img=np.array(test_img) 
test_img=preprocess_input(test_img)from keras.models import Modeldef vgg16_model(img_rows, img_cols, channel=1, num_classes=None):

    model = VGG16(weights='imagenet', include_top=True)

    model.layers.pop()

    model.outputs = [model.layers[-1].output]

    model.layers[-1].outbound_nodes = []

          x=Dense(num_classes, activation='softmax')(model.output)

    model=Model(model.input,x)#To set the first 8 layers to non-trainable (weights will not be updated)

          for layer in model.layers[:8]:

       layer.trainable = False# Learning rate is changed to 0.001
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])    return model

train_y=np.asarray(train['label'])

le = LabelEncoder()

train_y = le.fit_transform(train_y)

train_y=to_categorical(train_y)

train_y=np.array(train_y)from sklearn.model_selection import train_test_split
X_train, X_valid, Y_train, Y_valid=train_test_split(train_img,train_y,test_size=0.2, random_state=42)# Example to fine-tune on 3000 samples from Cifar10img_rows, img_cols = 224, 224 # Resolution of inputschannel = 3num_classes = 10 batch_size = 16 nb_epoch = 10# Load our modelmodel = vgg16_model(img_rows, img_cols, channel, num_classes)

model.summary()# Start Fine-tuningmodel.fit(X_train, Y_train,batch_size=batch_size,epochs=nb_epoch,shuffle=True,verbose=1,validation_data=(X_valid, Y_valid))# Make predictionspredictions_valid = model.predict(X_valid, batch_size=batch_size, verbose=1)# Cross-entropy loss scorescore = log_loss(Y_valid, predictions_valid)