from keras.layers import Embedding, TimeDistributed, RepeatVector, LSTM, concatenate , Input, Reshape, Dense
from keras.preprocessing.image import array_to_img, img_to_array, load_img
import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
from IPython.core.display import display, HTML
"""
# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text
 
filename = 'Flickr8k_text/Flickr8k.token.txt'
# load descriptions
doc = load_doc(filename)


# extract descriptions for images
def load_descriptions(doc):
	mapping = dict()
	# process lines
	for line in doc.split('\n'):
		# split line by white space
		tokens = line.split()
		if len(line) < 2:
			continue
		# take the first token as the image id, the rest as the description
		image_id, image_desc = tokens[0], tokens[1:]
		# remove filename from image id
		image_id = image_id.split('.')[0]
		# convert description tokens back to string
		image_desc = ' '.join(image_desc)
		# store the first description for each image
		if image_id not in mapping:
			mapping[image_id] = image_desc
	return mapping
 
# parse descriptions
descriptions = load_descriptions(doc)
print('Loaded: %d ' % len(descriptions))


"""

def img_feature():
#Length of longest sentence

    max_caption_len = 3
#Size of vocabulary 
    vocab_size = 3
# Load one screenshot for each word and turn them into digits 
    images = []
    for i in range(2):
        images.append(img_to_array(load_img('bd_logo1.png', target_size=(224, 224))))
    images = np.array(images, dtype=float) 
# Preprocess input for the VGG16 model
    images = preprocess_input(images)
    # Load the VGG16 model trained on imagenet and output the classification feature
    VGG = VGG16(weights='imagenet', include_top=True)
    # Extract the features from the image
    features = VGG.predict(images)
    return features

def simple(vgg_feature):  
#Turn start tokens into one-hot encoding
    html_input = np.array(
                [[[0., 0., 0.], #start
                 [0., 0., 0.],
                 [1., 0., 0.]],
                 [[0., 0., 0.], #start <HTML>Hello World!</HTML>
                 [1., 0., 0.],
                 [0., 1., 0.]]])
#Turn next word into one-hot encoding

    next_words = np.array(
                [[0., 1., 0.], # <HTML>Hello World!</HTML>
                 [0., 0., 1.]]) # end
    #Load the feature to the network, apply a dense layer, and repeat the vector

    vgg_feature = Input(shape=(1000,))

    vgg_feature_dense = Dense(5)(vgg_feature)

    vgg_feature_repeat = RepeatVector(max_caption_len)(vgg_feature_dense)
    # shape (3,5)

    
    # Extract information from the input seqence 

    language_input = Input(shape=(vocab_size, vocab_size)) #timestep,

    language_model = LSTM(5, return_sequences=True)(language_input)#


    
    # Concatenate the information from the image and the input

    decoder = concatenate([vgg_feature_repeat, language_model])
    # Extract information from the concatenated output

    decoder = LSTM(5, return_sequences=False)(decoder)
    # Predict which word comes next

    decoder_output = Dense(vocab_size, activation='softmax')(decoder)
    # Compile and run the neural network

    model = Model(inputs=[vgg_feature, language_input], outputs=decoder_output)

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    # Train the neural network

    model.fit([features, html_input], next_words, batch_size=2, shuffle=False, epochs=1000)
    return model
if __name__ == '__main__':
    #features=img_feature()
    #model=simple(features)#truan
    start_token = [1., 0., 0.] # start
    sentence = np.zeros((1, 3, 3)) # [[0,0,0], [0,0,0], [0,0,0]]
    sentence[0][2] = start_token # place start in empty sentence
    print(sentence)
    # Making the first prediction with the start token
    """
    second_word = model.predict([np.array([features[1]]), sentence])
    # Put the second word in the sentence and make the final prediction
    sentence[0][1] = start_token
    sentence[0][2] = np.round(second_word)
    third_word = model.predict([np.array([features[1]]), sentence])
    sentence[0][0] = start_token
    sentence[0][1] = np.round(second_word)
    sentence[0][2] = np.round(third_word)
    # Transform our one-hot predictions into the final tokens
    vocabulary = ["start", "<HTML><center><H1>Hello World!</H1><center></HTML>", "end"]
    html = ""
    for i in sentence[0]:
        html += vocabulary[np.argmax(i)] + ' '
    from IPython.core.display import display, HTML
    display(HTML(html[6:49]))
    """

    #a=[[3,3,3,5,6],[3,5,4,5,4],[9,5,4,3,3]]
    #b=[[1,1,2,2,6],[3,1,1,5,4],[1,1,1,1,1]]

    #concatenate([a, b])