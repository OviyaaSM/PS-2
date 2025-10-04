import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img

# -----------------------------
# U-Net Model Definition
# -----------------------------
def build_unet(input_size=(256, 256, 3)):
    inputs = layers.Input(input_size)

    # Encoder
    c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D(2)(c1)

    c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D(2)(c2)

    c3 = layers.Conv2D(256, 3, activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, 3, activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D(2)(c3)

    c4 = layers.Conv2D(512, 3, activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(512, 3, activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D(2)(c4)

    # Bottleneck
    c5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(c5)

    # Decoder
    u6 = layers.Conv2DTranspose(512, 2, strides=2, padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(512, 3, activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(512, 3, activation='relu', padding='same')(c6)

    u7 = layers.Conv2DTranspose(256, 2, strides=2, padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(256, 3, activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(256, 3, activation='relu', padding='same')(c7)

    u8 = layers.Conv2DTranspose(128, 2, strides=2, padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(128, 3, activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(128, 3, activation='relu', padding='same')(c8)

    u9 = layers.Conv2DTranspose(64, 2, strides=2, padding='same')(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv2D(64, 3, activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(64, 3, activation='relu', padding='same')(c9)

    outputs = layers.Conv2D(3, 1, activation='sigmoid')(c9)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model


# -----------------------------
# Dataset Loader
# -----------------------------
def load_dataset(input_dir, target_dir, size=(256, 256)):
    X, Y = [], []
    for filename in os.listdir(input_dir):
        if filename in os.listdir(target_dir):  # ensure pair exists
            input_img = load_img(os.path.join(input_dir, filename), target_size=size)
            target_img = load_img(os.path.join(target_dir, filename), target_size=size)
            X.append(img_to_array(input_img) / 255.0)
            Y.append(img_to_array(target_img) / 255.0)
    return np.array(X), np.array(Y)


# -----------------------------
# Train Model
# -----------------------------
def train_model(input_path, target_path, model_path="unet_weights.h5", epochs=25, batch_size=8):
    X, Y = load_dataset(input_path, target_path)
    print("Dataset loaded:", X.shape, Y.shape)

    model = build_unet()
    model.compile(optimizer='adam', loss='mae', metrics=['accuracy'])

    model.fit(
        X, Y,
        validation_split=0.1,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1
    )

    model.save(model_path)
    print(f"✅ Model saved at {model_path}")
    return model


# -----------------------------
# Load Trained Model
# -----------------------------
def load_trained_model(model_path="unet_weights.h5"):
    model = build_unet()
    model.load_weights(model_path)
    return model


# -----------------------------
# Enhance Single Image
# -----------------------------
def enhance_image(model, image_path, output_path="enhanced_output.jpg", size=(256, 256)):
    img = load_img(image_path, target_size=size)
    img_array = img_to_array(img) / 255.0
    pred = model.predict(np.expand_dims(img_array, axis=0))[0]
    enhanced_img = array_to_img(pred)
    enhanced_img.save(output_path)
    print(f"✅ Enhanced image saved at {output_path}")
    return output_path
