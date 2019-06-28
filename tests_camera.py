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


def similarity(frame1, frame2):
    # Transforme image to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    hist1 = cv2.calcHist([gray1],[0],None,[256],[0,256])
    hist2 = cv2.calcHist([gray2],[0],None,[256],[0,256])
    comp = cv2.compareHist(hist1, hist2, 0)
    return comp


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


os.chdir('C:/Users/frede/.spyder-py3/PROJET')
model_vgg16 = VGG16()
# remove the classifier layers
model_vgg16 = Model(inputs=model_vgg16.inputs, outputs=model_vgg16.layers[-2].output)

with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
    model = load_model('model_19.h5')
    
# pre-define the max sequence length (from training)
max_length = 34


font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomSideOfText       = (200,350)
topLeftCornerOfText    = (350, 10)
fontScale              = 1
fontColor              = (0,255,70)
lineType               = 2

cap = cv2.VideoCapture(0)
cap.set(3,224)
cap.set(4,224)
frameRate = cap.get(5)
_, prev_frame = cap.read()

compteur = 1
justCalculated = False

while(True):
    # Capture frame-by-frame
    _, frame = cap.read()
    # Our operations on the frame come here
    similar = similarity(frame, prev_frame)
    #frame_features = extract_features(frame)
    cv2.putText(frame,str(similar),
                topLeftCornerOfText, 
                font, 
                fontScale/2,
                fontColor,
                1)
    
    if (similar < 0.8 and justCalculated == False):
        justCalculated = True
        # Put prediction HERE
        
        cv2.putText(frame,str(""prediction""),
                topLeftCornerOfText, 
                font, 
                fontScale/2,
                fontColor,
                1)
        

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if (compteur % 15 == 0):
        prev_frame = frame
        justCalculated = False
    compteur += 1
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()