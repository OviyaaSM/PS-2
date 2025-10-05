# backend/unet_model.py
import os
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
import tensorflow as tf

def build_unet(input_size=(256,256,3)):
    inputs = layers.Input(input_size)
    # Encoder
    c1 = layers.Conv2D(64,3,activation='relu',padding='same')(inputs)
    c1 = layers.Conv2D(64,3,activation='relu',padding='same')(c1)
    p1 = layers.MaxPooling2D(2)(c1)
    c2 = layers.Conv2D(128,3,activation='relu',padding='same')(p1)
    c2 = layers.Conv2D(128,3,activation='relu',padding='same')(c2)
    p2 = layers.MaxPooling2D(2)(c2)
    c3 = layers.Conv2D(256,3,activation='relu',padding='same')(p2)
    c3 = layers.Conv2D(256,3,activation='relu',padding='same')(c3)
    p3 = layers.MaxPooling2D(2)(c3)
    c4 = layers.Conv2D(512,3,activation='relu',padding='same')(p3)
    c4 = layers.Conv2D(512,3,activation='relu',padding='same')(c4)
    p4 = layers.MaxPooling2D(2)(c4)
    c5 = layers.Conv2D(1024,3,activation='relu',padding='same')(p4)
    c5 = layers.Conv2D(1024,3,activation='relu',padding='same')(c5)
    u6 = layers.Conv2DTranspose(512, 2, strides=2, padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(512,3,activation='relu',padding='same')(u6)
    c6 = layers.Conv2D(512,3,activation='relu',padding='same')(c6)
    u7 = layers.Conv2DTranspose(256,2,strides=2,padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(256,3,activation='relu',padding='same')(u7)
    c7 = layers.Conv2D(256,3,activation='relu',padding='same')(c7)
    u8 = layers.Conv2DTranspose(128,2,strides=2,padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(128,3,activation='relu',padding='same')(u8)
    c8 = layers.Conv2D(128,3,activation='relu',padding='same')(c8)
    u9 = layers.Conv2DTranspose(64,2,strides=2,padding='same')(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv2D(64,3,activation='relu',padding='same')(u9)
    c9 = layers.Conv2D(64,3,activation='relu',padding='same')(c9)
    outputs = layers.Conv2D(3,1,activation='sigmoid')(c9)
    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model

def load_trained_model(weights_path="unet_weights.h5"):
    model = build_unet()
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"{weights_path} not found")
    model.load_weights(weights_path)
    return model

def enhance_image(model, input_path, output_path="enhanced_unet.jpg", size=(256,256)):
    img = load_img(input_path, target_size=size)
    arr = img_to_array(img)/255.0
    pred = model.predict(np.expand_dims(arr, axis=0))[0]
    out = array_to_img(pred)
    out.save(output_path)
    return output_path
