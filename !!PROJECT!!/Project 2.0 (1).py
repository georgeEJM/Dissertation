#!/usr/bin/env python
# coding: utf-8

# In[363]:


#George Monk Year 3 Project
#University of Lincoln


#Importing the required libraries
import matplotlib.pyplot as plt # For plotting images
import numpy as np # Mathematical requirement
import cv2 as cv # Reading in images
from PIL import Image # Image handling
from skimage import color # Efficient colour conversion
import glob # For reading in files


# In[364]:


# Keras imports
from keras import layers
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model 
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, LayerNormalization, Activation, Reshape
from tensorflow.keras.utils import to_categorical
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv2DTranspose 
from keras.layers import LeakyReLU, Input, Lambda, Input, Concatenate, RepeatVector, GlobalAveragePooling2D
from keras.utils import np_utils # Converts y_train and test to categorical values (softmax)
from keras.datasets import cifar10
from keras.optimizers import Adam


# In[365]:


# CHecking for GPU support
import tensorflow as tf # Importing tensorflow to GPU testing availability
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU'))) # Prints number of GPUS available
print(tf.__version__)
gpu = len(tf.config.list_physical_devices('GPU'))>0 # If there is a GPU ready
print("GPU is", "available" if gpu else "NOT AVAILABLE") # State if there is a GPU ready or not

for gpu in tf.config.experimental.list_physical_devices('GPU'): # Sets dynamic memory growth for each GPU in use. 
    tf.config.experimental.set_memory_growth(gpu, True) # This stops runtime initialization from using all of the memory.


# In[366]:


(X_train, y_train), (X_test, y_test) = cifar10.load_data() # Loading in Cifar10 data
y_train_softmax = np_utils.to_categorical(y_train, 10) # Gets probabilistic variants of y_train and y_test
y_test_softmax  = np_utils.to_categorical(y_test, 10)


# In[367]:


plt.imshow(X_train[0]) # Example Cifar10 image


# In[368]:


flickr_files = glob.glob('F:/Datasets/Flickr/test/*.png') # Loading in Flickr files. You will have to change this.

arr = [] # To put them in

width = int(512 * 25 / 100) # Reducing them to 128 by 128
height = int(512 * 25 / 100)
dsize = (width, height)

for file in files: # Reads in files converts them to correct colour format and resizes them. 
    image = cv.imread(file)
    imgrgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    imgrez = cv.resize(imgrgb, dsize)
    arr.append(imgrez) # Adds image to the array


# In[369]:


flickr = np.array(arr) # Gets numpy array from the array
flickr_train = flickr[0:1050] # Gets training and test sets for the flickr faces dataset
flickr_test = flickr[1050:]


# In[319]:


plt.imshow(flickr[0]) # Example flickr faces image


# In[320]:


unrelated_files = glob.glob('F:/Datasets/Personal/*.png') # Loading in Flickr files

other_arr = [] # To put them in

width = int(512 * 25 / 100) # Reducing them to 128 by 128
height = int(512 * 25 / 100)
dsize = (width, height)

for file in unrelated_files: # Reads in files converts them to correct colour format and resizes them. 
    image = cv.imread(file)
    imgrgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    imgrez = cv.resize(imgrgb, dsize)
    other_arr.append(imgrez) # Adds image to the array


# In[321]:


unrelated = np.array(other_arr) # Creates array of unrelated images


# In[322]:


plt.imshow(unrelated[0])


# In[361]:


def Normalize(x): # Normalization formula - not used
    min = np.min(x)
    max = np.max(x)
    
    range = max - min
    return [(a - min) / range for a in x]

def Interval_Mapping(image, from_min, from_max, to_min, to_max): # Interval mapping
    # map values from [from_min, from_max] to [to_min, to_max]
    # image: input array
    from_range = from_max - from_min
    to_range = to_max - to_min
    scaled = np.array((image - from_min) / float(from_range), dtype=float)
    return to_min + (scaled * to_range)

def Visualize_Data(images, categories, class_names, x, y): # Visualises data alognside predicted class names
    fig = plt.figure(figsize=(10, 10))
    fig.patch.set_facecolor('white')
    for i in range(x * y):
        plt.subplot(x, y, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(images[i])
        class_index = categories[i].argmax()
        plt.xlabel(class_names[class_index])
    plt.show()


# In[324]:


def ColourConvert(X_train, X_test, dim): # Used colour conversion and normalization method
    
    numEl_train = X_train.shape[0] # Gets number of elements
    X_L = np.empty(shape = [numEl_train, dim, dim]) # To create empty arrays using the provided dimensions
    X_AB = np.empty(shape = [numEl_train, dim, dim, 2])

    numEl_test = X_test.shape[0] # Does so again for test data
    X_L_TEST = np.empty(shape = [numEl_test, dim, dim])
    X_AB_TEST = np.empty(shape = [numEl_test, dim, dim, 2])

    for img in range(X_L.shape[0]): # For every image
        labImg = Image.fromarray(X_train[img], 'RGB') # Gets an actual image type from np array
        labImg = color.rgb2lab(labImg) # Converts it to L*a*b* format
        L = labImg[:,:,0] # Splits L* and a*b* channels
        AB = labImg[:,:,1:]
        greyImg = np.asarray(L, dtype="int32") # Creates numpy arrays from those values
        colChannels = np.asarray(AB, dtype="int32")
        X_L[img] = greyImg # Puts them into separate arrays       
        X_AB[img] = colChannels 

    X_L =  np.expand_dims(X_L, -1) # Reduces dimension for Keras formatting
    for img in range(X_L_TEST.shape[0]): # Repeats with test data
        labImg = Image.fromarray(X_test[img], 'RGB')
        labImg = color.rgb2lab(labImg)

        L = labImg[:,:,0]
        AB = labImg[:,:,1:]

        greyImg = np.asarray(L, dtype="int32")
        colChannels = np.asarray(AB, dtype="int32")
        X_L_TEST[img] = greyImg       
        X_AB_TEST[img] = colChannels 

    X_L_TEST =  np.expand_dims(X_L_TEST, -1)
    
    X_L_n = X_L.astype("float32") # Converts data to float

    X_L_n = Interval_Mapping(X_L_n, 0, 100, -1.0, 1.0) # Normalizes it between given range, and repeats for all other created arrays
    X_L_TEST_n = X_L_TEST.astype("float32")
    X_L_TEST_n = Interval_Mapping(X_L_TEST_n, 0, 100, -1.0, 1.0)

    X_AB_n = X_AB.astype("float32")
    X_AB_n = Interval_Mapping(X_AB_n, -128, 128, -1.0, 1.0)

    X_AB_TEST_n = X_AB_TEST.astype("float32")

    X_AB_TEST_n = Interval_Mapping(X_AB_TEST_n, -128, 128, -1.0, 1.0)
    return X_L_n, X_AB_n, X_L_TEST_n, X_AB_TEST_n # Gets data back


# In[325]:


cifar_L, cifar_AB, cifar_L_Test, cifar_AB_Test = ColourConvert(X_train, X_test, 32) # Gets L* and a*b* channels for both training and test data
flickr_L, flickr_AB, flickr_L_Test, flickr_AB_Test = ColourConvert(flickr_train, flickr_test, 128)


# In[326]:


def ColourConvert_TrainOnly(X_train): # The same method as above, but only outputs 'training' data
    
    numEl_train = X_train.shape[0] #X_TRAIN
    X_L = np.empty(shape = [numEl_train, 128, 128])
    X_AB = np.empty(shape = [numEl_train, 128, 128, 2])

    for img in range(X_L.shape[0]):
        labImg = Image.fromarray(X_train[img], 'RGB')
        labImg = color.rgb2lab(labImg)
        L = labImg[:,:,0]
        AB = labImg[:,:,1:]
        greyImg = np.asarray(L, dtype="int32")
        colChannels = np.asarray(AB, dtype="int32")
        X_L[img] = greyImg       
        X_AB[img] = colChannels 

    X_L =  np.expand_dims(X_L, -1)
    X_L_n = X_L.astype("float32")
    X_L_n = interval_mapping(X_L_n, 0, 100, -1.0, 1.0)
    
    X_AB_n = X_AB.astype("float32")
    X_AB_n = interval_mapping(X_AB_n, -128, 128, -1.0, 1.0)

    return X_L_n, X_AB_n


# In[327]:


unrelated_L, _ = ColourConvert_TrainOnly(unrelated)


# In[328]:


print(cifar_L.shape)


# In[329]:


plt.imshow(X_train[0]) # Demonstrates that no colour or image information is lost this way


# In[330]:


testImg_nn = interval_mapping(cifar_L[0], -1, 1, 0, 100) # Converts image back
testResult_nn = interval_mapping(cifar_AB[0], -1, 1, -128, 128) 
plt.imshow(color.lab2rgb(np.dstack((testImg_nn, testResult_nn))))


# In[331]:


data_augmentation = keras.Sequential( # Data augmentation
    [
        layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        layers.experimental.preprocessing.RandomRotation(0.1),
        layers.experimental.preprocessing.RandomZoom(0.1),
    ]
)


# In[389]:


# COLOURISER W/ FEATURES - CIFAR10

image_Input = Input(shape=(32,32,1)) # Inputs
feature_Input = Input(shape=(10))

MrFirst = data_augmentation(image_Input) # Applies data augmentation declared above to input
MrFirst = Conv2D(64, (3,3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(MrFirst) # Actual convolution
MrFirst = BatchNormalization()(MrFirst) # Applies batch normalization
MrFirst = Dropout(0.185)(MrFirst) # Adds dropout
MrFirst = Conv2D(128, (3,3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(MrFirst)
MrFirst = BatchNormalization()(MrFirst)
MrFirst = Dropout(0.185)(MrFirst)

Auto1 = Conv2D(128, (1,1), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(MrFirst) # First autoencoder stage
Auto1 = BatchNormalization()(Auto1)
Auto1 = Dropout(0.185)(Auto1)
Auto1 = Conv2D(128, (3,3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(Auto1)
Auto1 = BatchNormalization()(Auto1)
Auto1 = Dropout(0.185)(Auto1)

Auto2 = Conv2D(256, (1,1), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(Auto1) # Second autoencoder stage
Auto2 = BatchNormalization()(Auto2)
Auto2 = Dropout(0.185)(Auto2)
Auto2 = Conv2D(256, (3,3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(Auto2)
Auto2 = BatchNormalization()(Auto2)
Auto2 = Dropout(0.185)(Auto2)

Auto3 = Conv2D(384, (1,1), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(Auto2) # Third autoencoder stage
Auto3 = BatchNormalization()(Auto3)
Auto3 = Dropout(0.185)(Auto3)
Auto3 = Conv2D(384, (3,3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(Auto3)
Auto3 = BatchNormalization()(Auto3)
Auto3 = Dropout(0.185)(Auto3)

Auto4 = Conv2D(384, (1,1), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(Auto3) # Fourth autoencoder stage
Auto4 = BatchNormalization()(Auto4)
Auto4 = Dropout(0.185)(Auto4)
Auto4 = Conv2D(384, (3,3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(Auto4)
Auto4 = BatchNormalization()(Auto4)
Auto4 = Dropout(0.185)(Auto4)

concat_shape = (np.uint32(Auto4.shape[1]), np.uint32(Auto4.shape[2]),np.uint32(FeatureInput.shape[-1])) # Gets desired shape for
                                                                                                        # feature inclusion

img_features = RepeatVector(concat_shape[0]*concat_shape[1])(feature_Input) # Repeats the features into correct size
img_features = Reshape(concat_shape)(img_features) # Shapes them to correct dimensions and size
img_features = Conv2D(384, (3,3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(img_features) # Simple convolution
img_features = BatchNormalization()(img_features)
img_features = Dropout(0.5)(img_features) # High dropout to not overuse them

MrWorldWide = Concatenate()([Auto4, img_features]) # Concatenates them with past autoencoder stage

Deco1 = Conv2DTranspose(384, (3,3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(MrWorldWide) # Starts decoding
Deco1 = BatchNormalization()(Deco1)
Deco1 = Dropout(0.185)(Deco1)
Deco1 = Concatenate()([Deco1, Auto4])
Deco1 = Conv2DTranspose(384, (1,1), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(Deco1)
Deco1 = BatchNormalization()(Deco1)
Deco1 = Dropout(0.185)(Deco1)

Deco2 = Conv2DTranspose(384, (3,3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(Deco1) # Decoder block 2
Deco2 = BatchNormalization()(Deco2)
Deco2 = Dropout(0.185)(Deco2)
Deco2 = Concatenate()([Deco2, Auto3])
Deco2 = Conv2DTranspose(384, (1,1), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(Deco2) 
Deco2 = BatchNormalization()(Deco2)
Deco2 = Dropout(0.185)(Deco2)

Deco3 = Conv2DTranspose(256, (3,3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(Deco2) # Decoder block 3
Deco3 = BatchNormalization()(Deco3)
Deco3 = Dropout(0.225)(Deco3)
Deco3 = Concatenate()([Deco3, Auto2])
Deco3 = Conv2DTranspose(256, (1,1), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(Deco3)
Deco3 = BatchNormalization()(Deco3)
Deco3 = Dropout(0.185)(Deco3)

Deco4 = Conv2DTranspose(128, (3,3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(Deco3) # Decoder block 4
Deco4 = BatchNormalization()(Deco4)
Deco4 = Dropout(0.185)(Deco4)
Deco4 = Concatenate()([Deco4, Auto1])
Deco4 = Conv2DTranspose(127, (1,1), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(Deco4)
Deco4 = BatchNormalization()(Deco4)
Deco4 = Dropout(0.185)(Deco4)

MrMerger = Concatenate()([Deco4, image_Input]) # Merges decoder with initial input

Split = Lambda( lambda x: tf.split(x,num_or_size_splits=2,axis=3))(MrMerger) # Splits it in two

# Uses two splits to perform operations multidimensionally
Left = Conv2DTranspose(128, (3,3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(Split[0]) 
Left = BatchNormalization()(Left)
Left = Dropout(0.185)(Left)

Left = Conv2DTranspose(64, (3,3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(Left)
Left = BatchNormalization()(Left)
Left = Dropout(0.185)(Left)

Right = Conv2DTranspose(128, (3,3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(Split[1])
Right = BatchNormalization()(Right)
Right = Dropout(0.185)(Right)

Right = Conv2DTranspose(64, (3,3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(Right)
Right = BatchNormalization()(Right)
Right = Dropout(0.185)(Right)

MrMiddle = Concatenate()([Left, Right]) # Concatenates the split back together

MrMiddle = Conv2DTranspose(64, (3,3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(MrMiddle) # Final convolutions
MrMiddle = BatchNormalization()(MrMiddle)
MrMiddle = Dropout(0.185)(MrMiddle)
MrMiddle = Conv2DTranspose(64, (1,1), strides=1, activation='softmax', padding='same')(MrMiddle) # Softmax for probability
MrMiddle = BatchNormalization()(MrMiddle)
MrMiddle = Dropout(0.4)(MrMiddle)
MrPredictor = Conv2DTranspose(2, (3,3), strides=1, activation='tanh', padding='same')(MrMiddle) # Output


cifar_Colouriser = Model([image_Input, feature_Input], MrPredictor) # Produces model
cifar_Colouriser.summary() # Provides summary of model


# In[390]:


# COLOURISER W/OUT FEATURES- CIFAR10

image_Input = Input(shape=(32,32,1)) # Exactly the same as before, but without features

MrFirst = data_augmentation(ClassInput)
MrFirst = Conv2D(64, (3,3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(MrFirst)
MrFirst = BatchNormalization()(MrFirst)
MrFirst = Dropout(0.185)(MrFirst)
MrFirst = Conv2D(128, (3,3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(MrFirst)
MrFirst = BatchNormalization()(MrFirst)
MrFirst = Dropout(0.185)(MrFirst)

Auto1 = Conv2D(128, (1,1), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(MrFirst)
Auto1 = BatchNormalization()(Auto1)
Auto1 = Dropout(0.185)(Auto1)
Auto1 = Conv2D(128, (3,3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(Auto1)
Auto1 = BatchNormalization()(Auto1)
Auto1 = Dropout(0.185)(Auto1)

Auto2 = Conv2D(256, (1,1), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(Auto1)
Auto2 = BatchNormalization()(Auto2)
Auto2 = Dropout(0.185)(Auto2)
Auto2 = Conv2D(256, (3,3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(Auto2)
Auto2 = BatchNormalization()(Auto2)
Auto2 = Dropout(0.185)(Auto2)

Auto3 = Conv2D(384, (1,1), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(Auto2)
Auto3 = BatchNormalization()(Auto3)
Auto3 = Dropout(0.185)(Auto3)
Auto3 = Conv2D(384, (3,3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(Auto3)
Auto3 = BatchNormalization()(Auto3)
Auto3 = Dropout(0.185)(Auto3)

Auto4 = Conv2D(384, (1,1), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(Auto3)
Auto4 = BatchNormalization()(Auto4)
Auto4 = Dropout(0.185)(Auto4)
Auto4 = Conv2D(384, (3,3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(Auto4)
Auto4 = BatchNormalization()(Auto4)
Auto4 = Dropout(0.185)(Auto4)

Deco1 = Conv2DTranspose(384, (3,3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(Auto4)
Deco1 = BatchNormalization()(Deco1)
Deco1 = Dropout(0.185)(Deco1)
Deco1 = Concatenate()([Deco1, Auto4])
Deco1 = Conv2DTranspose(384, (1,1), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(Deco1)
Deco1 = BatchNormalization()(Deco1)
Deco1 = Dropout(0.185)(Deco1)

Deco2 = Conv2DTranspose(384, (3,3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(Deco1)
Deco2 = BatchNormalization()(Deco2)
Deco2 = Dropout(0.185)(Deco2)
Deco2 = Concatenate()([Deco2, Auto3])
Deco2 = Conv2DTranspose(384, (1,1), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(Deco2)
Deco2 = BatchNormalization()(Deco2)
Deco2 = Dropout(0.185)(Deco2)

Deco3 = Conv2DTranspose(256, (3,3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(Deco2)
Deco3 = BatchNormalization()(Deco3)
Deco3 = Dropout(0.225)(Deco3)
Deco3 = Concatenate()([Deco3, Auto2])
Deco3 = Conv2DTranspose(256, (1,1), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(Deco3)
Deco3 = BatchNormalization()(Deco3)
Deco3 = Dropout(0.185)(Deco3)

Deco4 = Conv2DTranspose(128, (3,3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(Deco3)
Deco4 = BatchNormalization()(Deco4)
Deco4 = Dropout(0.185)(Deco4)
Deco4 = Concatenate()([Deco4, Auto1])
Deco4 = Conv2DTranspose(127, (1,1), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(Deco4)
Deco4 = BatchNormalization()(Deco4)
Deco4 = Dropout(0.185)(Deco4)

MrMerger = Concatenate()([Deco4, ClassInput])

Split = Lambda( lambda x: tf.split(x,num_or_size_splits=2,axis=3))(MrMerger)

Left = Conv2DTranspose(128, (3,3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(Split[0])
Left = BatchNormalization()(Left)
Left = Dropout(0.185)(Left)

Left = Conv2DTranspose(64, (3,3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(Left)
Left = BatchNormalization()(Left)
Left = Dropout(0.185)(Left)

Right = Conv2DTranspose(128, (3,3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(Split[1])
Right = BatchNormalization()(Right)
Right = Dropout(0.185)(Right)

Right = Conv2DTranspose(64, (3,3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(Right)
Right = BatchNormalization()(Right)
Right = Dropout(0.185)(Right)

MrMiddle = Concatenate()([Left, Right])

MrMiddle = Conv2DTranspose(64, (3,3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(MrMiddle)
MrMiddle = BatchNormalization()(MrMiddle)
MrMiddle = Dropout(0.185)(MrMiddle)
MrMiddle = Conv2DTranspose(64, (1,1), strides=1, activation='softmax', padding='same')(MrMiddle)
MrMiddle = BatchNormalization()(MrMiddle)
MrMiddle = Dropout(0.4)(MrMiddle)
MrPredictor = Conv2DTranspose(2, (3,3), strides=1, activation='tanh', padding='same')(MrMiddle)


cifar_Colouriser_NoFeatures = Model(ClassInput, MrPredictor)
cifar_Colouriser_NoFeatures.summary()


# In[391]:


# COLOURISER - FLICKR FACES

flickr_Input = Input(shape=(128,128,1)) # Exactly the same as before, but with a different input size and without features

MrFirst = data_augmentation(flickr_Input)
MrFirst = Conv2D(64, (3,3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(MrFirst)
MrFirst = BatchNormalization()(MrFirst)
MrFirst = Dropout(0.185)(MrFirst)
MrFirst = Conv2D(128, (3,3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(MrFirst)
MrFirst = BatchNormalization()(MrFirst)
MrFirst = Dropout(0.185)(MrFirst)

Auto1 = Conv2D(128, (1,1), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(MrFirst)
Auto1 = BatchNormalization()(Auto1)
Auto1 = Dropout(0.185)(Auto1)
Auto1 = Conv2D(128, (3,3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(Auto1)
Auto1 = BatchNormalization()(Auto1)
Auto1 = Dropout(0.185)(Auto1)

Auto2 = Conv2D(256, (1,1), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(Auto1)
Auto2 = BatchNormalization()(Auto2)
Auto2 = Dropout(0.185)(Auto2)
Auto2 = Conv2D(256, (3,3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(Auto2)
Auto2 = BatchNormalization()(Auto2)
Auto2 = Dropout(0.185)(Auto2)

Auto3 = Conv2D(384, (1,1), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(Auto2)
Auto3 = BatchNormalization()(Auto3)
Auto3 = Dropout(0.185)(Auto3)
Auto3 = Conv2D(384, (3,3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(Auto3)
Auto3 = BatchNormalization()(Auto3)
Auto3 = Dropout(0.185)(Auto3)

Auto4 = Conv2D(384, (1,1), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(Auto3)
Auto4 = BatchNormalization()(Auto4)
Auto4 = Dropout(0.185)(Auto4)
Auto4 = Conv2D(384, (3,3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(Auto4)
Auto4 = BatchNormalization()(Auto4)
Auto4 = Dropout(0.185)(Auto4)

Deco1 = Conv2DTranspose(384, (3,3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(Auto4)
Deco1 = BatchNormalization()(Deco1)
Deco1 = Dropout(0.185)(Deco1)
Deco1 = Concatenate()([Deco1, Auto4])
Deco1 = Conv2DTranspose(384, (1,1), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(Deco1)
Deco1 = BatchNormalization()(Deco1)
Deco1 = Dropout(0.185)(Deco1)

Deco2 = Conv2DTranspose(384, (3,3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(Deco1)
Deco2 = BatchNormalization()(Deco2)
Deco2 = Dropout(0.185)(Deco2)
Deco2 = Concatenate()([Deco2, Auto3])
Deco2 = Conv2DTranspose(384, (1,1), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(Deco2)
Deco2 = BatchNormalization()(Deco2)
Deco2 = Dropout(0.185)(Deco2)

Deco3 = Conv2DTranspose(256, (3,3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(Deco2)
Deco3 = BatchNormalization()(Deco3)
Deco3 = Dropout(0.225)(Deco3)
Deco3 = Concatenate()([Deco3, Auto2])
Deco3 = Conv2DTranspose(256, (1,1), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(Deco3)
Deco3 = BatchNormalization()(Deco3)
Deco3 = Dropout(0.185)(Deco3)

Deco4 = Conv2DTranspose(128, (3,3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(Deco3)
Deco4 = BatchNormalization()(Deco4)
Deco4 = Dropout(0.185)(Deco4)
Deco4 = Concatenate()([Deco4, Auto1])
Deco4 = Conv2DTranspose(127, (1,1), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(Deco4)
Deco4 = BatchNormalization()(Deco4)
Deco4 = Dropout(0.185)(Deco4)

MrMerger = Concatenate()([Deco4, flickr_Input])

Split = Lambda( lambda x: tf.split(x,num_or_size_splits=2,axis=3))(MrMerger)

Left = Conv2DTranspose(128, (3,3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(Split[0])
Left = BatchNormalization()(Left)
Left = Dropout(0.185)(Left)

Left = Conv2DTranspose(64, (3,3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(Left)
Left = BatchNormalization()(Left)
Left = Dropout(0.185)(Left)

Right = Conv2DTranspose(128, (3,3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(Split[1])
Right = BatchNormalization()(Right)
Right = Dropout(0.185)(Right)

Right = Conv2DTranspose(64, (3,3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(Right)
Right = BatchNormalization()(Right)
Right = Dropout(0.185)(Right)

MrMiddle = Concatenate()([Left, Right])

MrMiddle = Conv2DTranspose(64, (3,3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(MrMiddle)
MrMiddle = BatchNormalization()(MrMiddle)
MrMiddle = Dropout(0.185)(MrMiddle)
MrMiddle = Conv2DTranspose(64, (1,1), strides=1, activation='softmax', padding='same')(MrMiddle)
MrMiddle = BatchNormalization()(MrMiddle)
MrMiddle = Dropout(0.4)(MrMiddle)
MrPredictor = Conv2DTranspose(2, (3,3), strides=1, activation='tanh', padding='same')(MrMiddle)

flickr_Colouriser = Model(flickr_Input, MrPredictor)
flickr_Colouriser.summary()


# In[392]:


# CLASSIFIER - CIFAR10

classifier_Input = Input(shape=(32,32,1)) # Slightly different and outdated network, but works effectively for classification

MrFirst = data_augmentation(classifier_Input)
MrFirst = Conv2D(64, (3,3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(MrFirst)
MrFirst = BatchNormalization()(MrFirst)
MrFirst = Dropout(0.185)(MrFirst)
MrFirst = Conv2D(128, (3,3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(MrFirst)
MrFirst = BatchNormalization()(MrFirst)
MrFirst = Dropout(0.185)(MrFirst)

Auto1 = Conv2D(128, (1,1), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(MrFirst)
Auto1 = BatchNormalization()(Auto1)
Auto1 = Dropout(0.185)(Auto1)
Auto1 = Conv2D(128, (3,3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(Auto1)
Auto1 = BatchNormalization()(Auto1)
Auto1 = Dropout(0.185)(Auto1)

Auto2 = Conv2D(256, (1,1), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(Auto1)
Auto2 = BatchNormalization()(Auto2)
Auto2 = Dropout(0.185)(Auto2)
Auto2 = Conv2D(256, (3,3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(Auto2)
Auto2 = BatchNormalization()(Auto2)
Auto2 = Dropout(0.185)(Auto2)

Auto3 = Conv2D(256, (1,1), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(Auto2)
Auto3 = BatchNormalization()(Auto3)
Auto3 = Dropout(0.185)(Auto3)
Auto3 = Conv2D(256, (3,3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(Auto3)
Auto3 = BatchNormalization()(Auto3)
Auto3 = Dropout(0.185)(Auto3)

Auto4 = Conv2D(384, (1,1), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(Auto3)
Auto4 = BatchNormalization()(Auto4)
Auto4 = Dropout(0.185)(Auto4)
Auto4 = Conv2D(384, (3,3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(Auto4)
Auto4 = BatchNormalization()(Auto4)
Auto4 = Dropout(0.185)(Auto4)

Deco1 = Conv2DTranspose(384, (1,1), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(Auto4)
Deco1 = BatchNormalization()(Deco1)
Deco1 = Dropout(0.185)(Deco1)
Deco1 = Concatenate()([Deco1, Auto4])
Deco1 = Conv2DTranspose(384, (3,3), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(Deco1)
Deco1 = BatchNormalization()(Deco1)
Deco1 = Dropout(0.185)(Deco1)

Deco2 = Conv2DTranspose(256, (1,1), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(Deco1)
Deco2 = BatchNormalization()(Deco2)
Deco2 = Dropout(0.185)(Deco2)
Deco2 = Concatenate()([Deco2, Auto3])
Deco2 = Conv2DTranspose(256, (3,3), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(Deco2)
Deco2 = BatchNormalization()(Deco2)
Deco2 = Dropout(0.185)(Deco2)

Deco3 = Conv2DTranspose(256, (1,1), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(Deco2)
Deco3 = BatchNormalization()(Deco3)
Deco3 = Dropout(0.185)(Deco3)
Deco3 = Concatenate()([Deco3, Auto2])
Deco3 = Conv2DTranspose(256, (3,3), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(Deco3)
Deco3 = BatchNormalization()(Deco3)
Deco3 = Dropout(0.185)(Deco3)

Deco4 = Conv2DTranspose(128, (1,1), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(Deco3)
Deco4 = BatchNormalization()(Deco4)
Deco4 = Dropout(0.185)(Deco4)
Deco4 = Concatenate()([Deco4, Auto1])
Deco4 = Conv2DTranspose(127, (3,3), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(Deco4)
Deco4 = BatchNormalization()(Deco4)
Deco4 = Dropout(0.185)(Deco4)

MrMerger = Concatenate()([Deco4, classifier_Input])

Split = Lambda( lambda x: tf.split(x,num_or_size_splits=2,axis=3))(MrMerger)

Left = Conv2DTranspose(128, (3,3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(Split[0])
Left = BatchNormalization()(Left)
Left = Dropout(0.185)(Left)

Left = Conv2DTranspose(64, (3,3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(Left)
Left = BatchNormalization()(Left)
Left = Dropout(0.185)(Left)

Right = Conv2DTranspose(128, (3,3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(Split[1])
Right = BatchNormalization()(Right)
Right = Dropout(0.185)(Right)

Right = Conv2DTranspose(64, (3,3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(Right)
Right = BatchNormalization()(Right)
Right = Dropout(0.185)(Right)

MrMiddle = Concatenate()([Left, Right])

MrMiddle = Conv2DTranspose(64, (3,3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(MrMiddle)
MrMiddle = BatchNormalization()(MrMiddle)
MrMiddle = Dropout(0.185)(MrMiddle)
MrMiddle = Conv2DTranspose(64, (3,3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(MrMiddle)
MrMiddle = BatchNormalization()(MrMiddle)
Hello = GlobalAveragePooling2D()(MrMiddle)
Hello = Dense(500, activation=LeakyReLU(alpha=0.2))(Hello)
Hello = Dropout(0.5)(Hello)
Hello = Dense(10, activation = 'softmax')(Hello)

cifar_Classifier = Model(classifier_Input, Hello)
cifar_Classifier.summary()


# In[337]:


def TrainModel(model, x, y, x_test, y_test, epochs, batch_size): # Function to train model
    #model.compile(optimizer='adam', loss='mse', metrics = ['accuracy']) # Compiles model (if needed)
    model.fit(x,y, epochs=epochs, batch_size=batch_size, validation_split=0.2)  # Starts to train
    scores = model.evaluate(x_test, y_test, verbose=0) # Evaluates on test set when training is complete
    print("Accuracy: %.2f%%" % (scores[1]*100)) # Prints accuracy
    return model # Returns model

def TrainModelWithFeatures(colouriser, classifier, x, y, x_test, y_test, epochs, batch_size): # Same as above but with features
    predictions_for_train = classifier.predict(X_L_n)
    predictions_for_test = classifier.predict(X_L_TEST_n)
    #model.compile(optimizer='adam', loss='mse', metrics = ['accuracy'])
    model.fit([x, predictions_for_train],y, epochs=epochs, batch_size=batch_size, validation_split=0.2) 
    scores = model.evaluate([x_test, predictions_for_test], y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))
    return model


# In[371]:


TrainModel(flickr_Colouriser, flickr_L, flickr_AB, flickr_L_Test, flickr_AB_Test, 1, 4)


# In[393]:


cifar_Classifier.load_weights('Classification_Weights') # Loads saved weights
cifar_Colouriser_NoFeatures.load_weights('Cifar_Colourisation_Weights')
cifar_Colouriser.load_weights('Cifar_Colourisation_Weights_Features')
flickr_Colouriser.load_weights('Flickr_Colourisation_Weights')

cifar_Classifier.compile(optimizer='adam', loss='mse', metrics = ['accuracy']) # Compile every model
cifar_Colouriser_NoFeatures.compile(optimizer='adam', loss='mse', metrics = ['accuracy'])
cifar_Colouriser.compile(optimizer='adam', loss='mse', metrics = ['accuracy'])
flickr_Colouriser.compile(optimizer='adam', loss='mse', metrics = ['accuracy'])


# In[394]:


cifar_Classifier_Scores = cifar_Classifier.evaluate(cifar_L_Test, y_test_softmax, verbose=1) # Evaluate each model
print("Accuracy: %.2f%%" % (cifar_Classifier_Scores[1]*100))


# In[395]:


cifar_Colouriser_NoFeatures_Scores = cifar_Colouriser_NoFeatures.evaluate(cifar_L_Test, cifar_AB_Test, verbose=1)
print("Accuracy: %.2f%%" % (cifar_Colouriser_NoFeatures_Scores[1]*100))


# In[396]:


predictions_for_test = cifar_Classifier.predict(cifar_L_Test)
cifar_Colouriser_Scores = cifar_Colouriser.evaluate([cifar_L_Test, predictions_for_test], cifar_AB_Test, verbose=1)
print("Accuracy: %.2f%%" % (cifar_Colouriser_Scores[1]*100))


# In[397]:


flickr_Colouriser_Scores = flickr_Colouriser.evaluate(flickr_L_Test, flickr_AB_Test, verbose=1)
print("Accuracy: %.2f%%" % (flickr_Colouriser_Scores[1]*100))


# In[344]:


def PredictSingleImage(image, colouriser): # Displays a single predicted image
    testResult = colouriser.predict(image) # predicts colour
    testImg = np.squeeze(image)
    testResult = np.squeeze(testResult[0]) # Reconverts to RGB
    testResult = testResult.astype("float32")
    testImg_nn = interval_mapping(testImg, -1, 1, 0, 100) 
    testResult_nn = interval_mapping(testResult, -1, 1, -128, 128) 
    temp = (color.lab2rgb(np.dstack((testImg_nn, testResult_nn))))
    plt.imshow(temp) # Shows result


def PredictSingleImageWithFeatures(image, colouriser, classifier): # Same but with classifier added
    classResult = classifier.predict(image)
    testResult = colouriser.predict([image, classResult])
    testImg = np.squeeze(image)
    testResult = np.squeeze(testResult[0])
    testResult = testResult.astype("float32")
    testImg_nn = interval_mapping(testImg, -1, 1, 0, 100) 
    testResult_nn = interval_mapping(testResult, -1, 1, -128, 128) 
    temp = (color.lab2rgb(np.dstack((testImg_nn, testResult_nn))))
    plt.imshow(temp)


# In[345]:


def displayImages(images, x, y, iplus): # Displays multiple images in subplot
    display = plt.figure(figsize=(10,10))
    display.patch.set_facecolor('white')
    for i in range(x * y):
        plt.subplot(x, y, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(images[i+iplus])
    plt.show
    
def displayPredictions(images, colouriser, x, y, iplus): # Converts multiple images and displays them all on one plot
    display = plt.figure(figsize=(10,10))
    display.patch.set_facecolor('white')
    for i in range(x * y):
        plt.subplot(x, y, i+1) # iplus is used to sift through images starting at specific value
        plt.xticks([])
        plt.yticks([])
        testImg = images[i+iplus:i+iplus+1]
        testResult = colouriser.predict(testImg)
        testImg = np.squeeze(testImg)
        testResult = np.squeeze(testResult[0])
        testResult = testResult.astype("float32")
        testImg_nn = interval_mapping(testImg, -1, 1, 0, 100) 
        testResult_nn = interval_mapping(testResult, -1, 1, -128, 128) 
        temp = (color.lab2rgb(np.dstack((testImg_nn, testResult_nn))))
        plt.imshow(temp)
    plt.show

def displayPredictionsWithFeatures(images, colouriser, classifier, x, y, iplus):
    display = plt.figure(figsize=(10,10))
    display.patch.set_facecolor('white')
    for i in range(x * y):
        plt.subplot(x, y, i+1)
        plt.xticks([])
        plt.yticks([])
        testImg = images[i+iplus:i+iplus+1]
        
        classResult = classifier.predict(testImg)
        
        testResult = colouriser.predict([testImg, classResult])
        testImg = np.squeeze(testImg)
        testResult = np.squeeze(testResult[0])
        testResult = testResult.astype("float32")
        testImg_nn = interval_mapping(testImg, -1, 1, 0, 100) 
        testResult_nn = interval_mapping(testResult, -1, 1, -128, 128) 
        temp = (color.lab2rgb(np.dstack((testImg_nn, testResult_nn))))
        plt.imshow(temp)
    plt.show


# In[362]:


Visualize_Data(X_test, predictions_for_test, y_test, 5, 5)


# In[346]:


img_to_use = 0
plt.imshow(X_test[img_to_use])


# In[347]:


PredictSingleImage(cifar_L_Test[img_to_use:img_to_use+1], cifar_Colouriser_NoFeatures)


# In[348]:


PredictSingleImageWithFeatures(cifar_L_Test[img_to_use:img_to_use+1], cifar_Colouriser, cifar_Classifier)


# In[349]:


plt.imshow(flickr_test[img_to_use])


# In[350]:


PredictSingleImage(flickr_L_Test[img_to_use:img_to_use+1], flickr_Colouriser)


# In[351]:


PredictSingleImage(unrelated_L[0:1], flickr_Colouriser)


# In[352]:


displayImages(X_test, 5, 5, 0)


# In[353]:


displayPredictions(cifar_L_Test, cifar_Colouriser_NoFeatures, 5, 5, 0)


# In[354]:


displayPredictionsWithFeatures(cifar_L_Test, cifar_Colouriser, cifar_Classifier, 5, 5, 0)


# In[355]:


displayImages(flickr_test, 5, 5, 0)


# In[356]:


displayPredictions(flickr_L_Test, flickr_Colouriser, 5, 5, 0)


# In[ ]:




