# -*- coding: utf-8 -*-
import cv2
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras.initializers import glorot_uniform
from pickle import load
from numpy import argmax
from tensorflow.keras.preprocessing.sequence import pad_sequences

def extract_features(frame):
	# extract features from frame
	resized_image = cv2.resize(frame, (224, 224)) 
	image = img_to_array(resized_image)
	# reshape data for the model
	image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
	# prepare the image for the VGG model
	image = preprocess_input(image)
	# get features
	features = model_vgg16.predict(image, verbose=0)
	return features

# map an integer to a word
def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None
 
# generate a description for an image
def generate_desc(model, tokenizer, photo, max_length):
	# seed the generation process
	in_text = 'startseq'
	# iterate over the whole length of the sequence
	for i in range(max_length):
		# integer encode input sequence
		sequence = tokenizer.texts_to_sequences([in_text])[0]
		# pad input
		sequence = pad_sequences([sequence], maxlen=max_length)
		# predict next word
		yhat = model.predict([photo,sequence], verbose=0)
		# convert probability to integer
		yhat = argmax(yhat)
		# map integer to word
		word = word_for_id(yhat, tokenizer)
		# stop if we cannot map the word
		if word is None:
			break
		# append as input for generating the next word
		in_text += ' ' + word
		# stop if we predict the end of the sequence
		if word == 'endseq':
			break
	return in_text


os.chdir('C:/Users/frede/.spyder-py3/PROJET')
model_vgg16 = VGG16()
# remove the classifier layers
model_vgg16 = Model(inputs=model_vgg16.inputs, outputs=model_vgg16.layers[-2].output)

with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
    model = load_model('model_19.h5')
    
    
tokenizer = load(open('tokenizer.pkl', 'rb'))
# pre-define the max sequence length (from training)
max_length = 34
cap = cv2.VideoCapture(0)
cap.set(3,224)
cap.set(4,224)
frameRate = cap.get(5)
_, prev_frame = cap.read()
while(True):
    # Capture frame-by-frame
    _, frame = cap.read()
    # Our operations on the frame come here
    #similarité = cv2.compareHist(frame, prev_frame, 1)
    # Our operations on the frame come here
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_features = extract_features(frame)
    # generate description
    description = generate_desc(model, tokenizer, photo, max_length)
    print(description)
    # Display the resulting frame
    cv2.imshow('frame',frame)
    prev_frame = frame
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()