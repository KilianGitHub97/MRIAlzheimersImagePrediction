# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 13:14:31 2021

@author: kilia
"""
#Libraries
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator


#setwd
PATH = "C:\\Users\\kilia\\Documents\\GitHub\\MRIAlzheimersImagePrediction\\code"
os.chdir(PATH)

#import personal module
import funcs as fun

#import data
TRAIN_DIRECTORY = "..\\data\\train"
TEST_DIRECTORY = "..\\data\\test"
IMG_WIDTH = 176
IMG_HEIGHT = 208
BATCH_SIZE = 50

#count files
N = fun.count_obs(TRAIN_DIRECTORY)

#get class names
NAMES = fun.get_class_names(TRAIN_DIRECTORY)

#check the balance of the Dataset
fun.balance_check(
    folder_path = TRAIN_DIRECTORY,
    class_names = NAMES,
    number_of_obs = N,
    normalize=True
    )

# #make test directory
# fun.make_test_folder(
#     test_directory = TEST_DIRECTORY,
#     train_directory = TRAIN_DIRECTORY,
#     classes = NAMES,
#     total_number_of_observations = N,
#     percentage_test_directory = 20 
#     )


#calc min batch size to get a remainer of 0
fun.calc_batch_size_no_remainer(
    observations = N,
    max_batch_size = 150,
    min_batch_size = 5
    )

#data augmentation
data_generation = ImageDataGenerator(
    validation_split=0.2,
    rescale=1./255
    )

#read in training data
train = data_generation.flow_from_directory(
    directory = TRAIN_DIRECTORY, 
    target_size = (IMG_WIDTH, IMG_HEIGHT),
    color_mode = "grayscale",
    class_mode = "categorical",
    batch_size = BATCH_SIZE,
    shuffle = True,
    subset = "training"
    )

#load validation data
validation = data_generation.flow_from_directory(
    directory = TRAIN_DIRECTORY, 
    target_size = (IMG_WIDTH, IMG_HEIGHT),
    color_mode = "grayscale",
    class_mode = "categorical",
    batch_size = BATCH_SIZE,
    shuffle = True,
    subset = "validation"
)

#load test data
test = data_generation.flow_from_directory(
    directory = TEST_DIRECTORY, 
    target_size = (IMG_WIDTH, IMG_HEIGHT),
    color_mode = "grayscale",
    class_mode = "categorical",
    batch_size = BATCH_SIZE,
    shuffle = True
    )

#plot some sample images for the categories
fun.show_sample_img(
    img_path = TRAIN_DIRECTORY,
    col_names = NAMES
    )

#draw baseline model
model = keras.Sequential([
    keras.layers.InputLayer(input_shape=(IMG_WIDTH, IMG_HEIGHT, 1)),
    keras.layers.Conv2D(32, (3, 3), padding="valid", activation="relu"),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(64, (3, 3), activation="relu"),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(128, (3, 3), activation="relu"),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation="softmax"),
    keras.layers.Dense(4)
    ], name = "baseline_model")

model.summary()

#define callback
callbacks = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
    tf.keras.callbacks.ProgbarLogger(count_mode="samples")
    ]

#compile model
model.compile(
    optimizer = "Adam",
    loss = keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics = ["accuracy"]
    )

#train model
model.fit(
    x = train,
    epochs = 1,
    batch_size = 2,
    callbacks = callbacks,
    validation_data = test,
    verbose=2
    )

from keras.applications.vgg16 import VGG16

model = keras.applications.VGG16(include_top=True, weights="imagenet")
base_input = model.layers[0].input
base_output = model.layers[-0].output
final_output = keras.layers.Dense(4)(base_output)
new_model = keras.Model(inputs=base_input, outputs=final_output)

new_model.compile(
    optimizer = "Adam",
    loss = keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics = ["accuracy"]
    )

new_model.fit(
    x = train,
    epochs = 1,
    batch_size = BATCH_SIZE,
    validation_data = test,
    verbose=2
    )

VGG16 = VGG16(
    weights="imagenet",
    include_top=True,
    classifier_activation="softmax"
    )

model.fit(
    x = train,
    epochs = 1,
    batch_size = BATCH_SIZE,
    callbacks = callbacks,
    validation_data = test,
    verbose=2
    )
model.summary()
