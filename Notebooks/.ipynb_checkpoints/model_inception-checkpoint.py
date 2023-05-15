
# TensorFlow CNN model
# From "Photometric redshifts from SDSS images using a Convolutional Neural Network" 
# by J.Pasquet et al. 2018

# Adaptation for tensorflow v2 

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, PReLU, AveragePooling2D, Concatenate, Dense, Input, Flatten


def inception(input, nbS1, nbS2, name, output_name, without_kernel_5=False):

    # First layer
    cv1_0 = Conv2D(filters=nbS1, kernel_size=1, padding='same')(input)
    prelu1_0 = PReLU()(cv1_0)
    
    if not(without_kernel_5):
        cv1_1 = Conv2D(filters=nbS1, kernel_size=1, padding='same')(input)
        prelu1_1 = PReLU()(cv1_1)
    
    cv1_2 = Conv2D(filters=nbS1, kernel_size=1, padding='same')(input)
    prelu1_2 = PReLU()(cv1_2)
    
    # Second layer
    cv2_0 = Conv2D(filters=nbS2, kernel_size=1, padding='same')(input)
    prelu2_0 = PReLU()(cv2_0)
    
    cv2_1 = Conv2D(filters=nbS2, kernel_size=3, padding='same')(prelu1_0)
    prelu2_1 = PReLU()(cv2_1)
    
    if not(without_kernel_5):
        cv2_2 = Conv2D(filters=nbS2, kernel_size=5, padding='same')(prelu1_1)
        prelu2_2 = PReLU()(cv2_2)
    
    pool0 = AveragePooling2D(pool_size=2, strides=1, padding='same')(prelu1_2)
    
    # Third layer (concatenation)
    if not(without_kernel_5):
        concat = Concatenate()([prelu2_0, prelu2_1, prelu2_2, pool0])
    else:
        concat = Concatenate()([prelu2_0, prelu2_1, pool0])
    
    return concat



def model_tf2(with_ebv = False):
    
    Image = Input(shape=(64, 64, 5))
    if with_ebv:
        reddening = Input(shape=(1))
    
    conv0 = Conv2D(filters=64, kernel_size=5, padding='same')(Image)
    prelu0 = PReLU()(conv0)
    pool0 = AveragePooling2D(pool_size=2, strides=2, padding='same')(prelu0)
    
    i0 = inception(pool0, 48, 64, name="I0_", output_name="INCEPTION0")
    
    i1 = inception(i0, 64, 92, name="I1_", output_name="INCEPTION1")
    pool1 =AveragePooling2D(pool_size=2, strides=2, padding='same')(i1)
    
    i2 = inception(pool1, 92, 128, name="I2_", output_name="INCEPTION2")
    
    i3 = inception(i2, 92, 128, name="I3_", output_name="INCEPTION3")
    pool2 =AveragePooling2D(pool_size=2, strides=2, padding='same')(i3)
    
    i4 = inception(pool2, 92,128, name="I4_", output_name="INCEPTION4", without_kernel_5=True)
    
    flatten = Flatten()(i4)
    if with_ebv:
        concat = Concatenate()([flatten, reddening])
        d0 = Dense(1024, activation='relu')(concat)
    else:
        d0 = Dense(1024, activation='relu')(flatten)
    d1 = Dense(1024, activation='relu')(d0)
    outputs = Dense(1)(d1)
   
    if with_ebv:
        model = Model(inputs=[Image, reddening], outputs=outputs, name="incept_model")
    else:
        model = Model(inputs=Image, outputs=outputs, name="incept_model")
        
    return model
