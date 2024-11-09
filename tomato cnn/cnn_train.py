# -*- coding: utf-8 -*-
"""
CNN for Tomato Leaf Disease Detection using TensorFlow's Keras
"""

import os
os.environ["PATH"] += os.pathsep + r"C:\Program Files\Graphviz\bin"

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

np.random.seed(1337)

# Initialize the CNN Model
model = Sequential()

# Add Convolutional and Pooling Layers
model.add(Conv2D(32, (3, 3), input_shape=(128, 128, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(8, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten and Add Dense Layers
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(units=10, activation='softmax'))

# Compile the Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Data Augmentation and Loading
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    'train', target_size=(128, 128), batch_size=64, class_mode='categorical')

test_set = test_datagen.flow_from_directory(
    'val', target_size=(128, 128), batch_size=64, class_mode='categorical')

# Calculate steps per epoch and validation steps dynamically
steps_per_epoch = training_set.samples // training_set.batch_size
validation_steps = test_set.samples // test_set.batch_size

# Train the Model
model.fit(
    training_set,
    steps_per_epoch=steps_per_epoch,
    epochs=40,  # Adjust epochs as needed
    validation_data=test_set,
    validation_steps=validation_steps)

# Save the Model Weights
model.save('tomato_leaf_disease_model.h5')
print('Saved trained model as tomato_leaf_disease_weights.h5')
