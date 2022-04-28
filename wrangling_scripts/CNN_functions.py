from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.applications.resnet50 import ResNet50

from tensorflow.keras.preprocessing import image

from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras.models import Sequential

from tqdm import tqdm

from bottleneck_features.extract_bottleneck_features import *
import cv2

import matplotlib.pyplot as plt
import numpy as np

import re

import csv

def face_detector(img_path):
    # extract pre-trained face detector
    face_cascade = cv2.CascadeClassifier('face_detector/haarcascade_frontalface_alt.xml')
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

def ResNet50_predict_labels(img_path):
    """
    Function that returns prediction vector for image located at img_path
    
    """
    ResNet50_model = ResNet50(weights='imagenet')
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))

def dog_detector(img_path):
    """
    Function that returns "True" if a dog is detected
    in the image stored at img_path

    Input:
    
    - img_path: string with path to image.

    """
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))

def Resnet50_load_model():
    """
    Function that creates model
    """

    Resnet50_model = Sequential()
    Resnet50_model.add(GlobalAveragePooling2D(input_shape=(1,1,2048)))
    Resnet50_model.add(Dense(133, activation='softmax'))
    Resnet50_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    Resnet50_model.load_weights('saved_models/weights.best.Resnet50.hdf5')

    return Resnet50_model

def read_dog_names(filename):
    dog_names = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for dog in reader:
            dog_names.append(dog)
    return dog_names[0]

def Resnet50_predict_breed(img_path):
    """
    Function that predicts the dog breed of an image using a
    convolutional neural network that has been trained using
    transfer learning.
    
    Input:
    
    - img_path: string containing path to image to analyze.
    
    Output:
    
    - breed: string with corresponding dog breed.
    
    """
       
    # Extract bottleneck features.
    bottleneck_features = extract_Resnet50(path_to_tensor(img_path))

    # Get dog names.
    dog_names = read_dog_names('data/dog_names.csv')

    # Load model.
    Resnet50_model = Resnet50_load_model()
    
    # Obtain predicted vector.
    predicted_vector = Resnet50_model.predict(bottleneck_features)
    
    # Obtain dog breed that is predicted by the model.
    breed = re.sub("ages/train/[0-9]*.","", dog_names[np.argmax(predicted_vector)])
    breed = breed.replace("_"," ")
    
    return breed

def Resnet50_full_algorithm(img_path):
    """
    Function that checks if an image is dog or human, and
    prints the corresponding breed using Resnet50 model.
    
    Input:
    
    - img_path: string containing path to image to analyze.
    
    """

    # STEP 1: Check if image is a dog, human, or other and set flag variables.

    output = ''

    output += "----------- STEP 1 ------------ \n"

    # Initialize flag variables.
    is_dog = False
    is_human = False

    # Perform checks.
    if dog_detector(img_path):
        output += "SUCCESS! A dog has been detected by the algorithm. \n"
        is_dog = True
    
    elif face_detector(img_path):
        output += "WARNING: A human has been detected, but the algorithm " \
                + "will still provide a dog breed. Have fun! \n"
        is_human = True
    
    else:
        output += "ERROR: Neither a human or dog has been detected by the algorithm. " + "Try again with a different image."
    
    # STEP 2: If success on previous step, return breed.

    # Check flag variables.
    if is_dog or is_human:
        output += "----------- STEP 2 ------------ \n"
    
        breed = Resnet50_predict_breed(img_path)
    
        output += "The breed of the image predicted by the algorithm is " + str(breed) + "."

    return output
