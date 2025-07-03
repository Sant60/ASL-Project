# =============================
# File: train_model.py
# =============================

import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam

# CONFIGURATION
data_path = "HandSignsDataset"  # Folder with subfolders: M, N, R, T, U, V, etc.
img_size = 224
batch_size = 32
epochs = 15

# DATA GENERATOR
datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)

train_data = datagen.flow_from_directory(
    data_path,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_data = datagen.flow_from_directory(
    data_path,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# MODEL DEFINITION
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
    MaxPooling2D(2, 2),
    Dropout(0.25),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_data.num_classes, activation='softmax')
])

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# TRAINING
model.fit(
    train_data,
    validation_data=val_data,
    epochs=epochs
)

# SAVE MODEL & LABELS
model.save("keras_model.h5")

labels = list(train_data.class_indices.keys())
with open("labels.txt", "w") as f:
    for label in labels:
        f.write(label + "\n")

print("Model and labels saved successfully.")
