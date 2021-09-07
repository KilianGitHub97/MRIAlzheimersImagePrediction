# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 13:14:31 2021

@author: kilia
"""
################################## Input #####################################
##############################################################################

#Libraries
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
from keras.constraints import max_norm
from keras.applications.vgg19 import VGG19

#setwd
PATH = "D:\\Bibliotheken\\Dokumente\\GitHub\\MRIAlzheimersImagePrediction\\code"
#PATH = "C:\\Users\\kilia\\Documents\\GitHub\\MRIAlzheimersImagePrediction\\code"
os.chdir(PATH)

#import personal module
import funcs as fun


############################# Global Variables ###############################
##############################################################################

#import data
TRAIN_DIRECTORY = "..\\data\\train"
TEST_DIRECTORY = "..\\data\\test"
IMG_WIDTH = 176
IMG_HEIGHT = 208
BATCH_SIZE = 32

#count files
N_TRAIN = fun.count_obs(TRAIN_DIRECTORY)
N_TEST = fun.count_obs(TEST_DIRECTORY)

#get class names
NAMES = fun.get_class_names(TRAIN_DIRECTORY)

#check the balance of the Dataset
fun.balance_check( #Train
    folder_path = TRAIN_DIRECTORY,
    class_names = NAMES,
    number_of_obs = N_TRAIN,
    normalize = True
    )
fun.balance_check( #Test
    folder_path = TEST_DIRECTORY,
    class_names = NAMES,
    number_of_obs = N_TEST,
    normalize = False
    )

#CAUTION, DO ONLY APPLY ONCE!!!!!!!!
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
    observations = N_TRAIN,
    max_batch_size = 150,
    min_batch_size = 5
    )


############################ Data Augmentation ###############################
##############################################################################

#data augmentation
data_generation_train = ImageDataGenerator(
    rescale = 1./255,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    fill_mode = "nearest",
    brightness_range = [0.8, 1.2] #1 = neutral
    )

data_generation_test = ImageDataGenerator(
    rescale=1./255
    )


######################## Load Data From Harddrive ############################
##############################################################################

#read in training data
train = data_generation_train.flow_from_directory(
    directory = TRAIN_DIRECTORY,
    target_size = (IMG_WIDTH, IMG_HEIGHT),
    color_mode = "rgb",
    class_mode = "categorical",
    batch_size = BATCH_SIZE,
    shuffle = True #,
    )

#load test data
test = data_generation_test.flow_from_directory(
    directory = TEST_DIRECTORY, 
    target_size = (IMG_WIDTH, IMG_HEIGHT),
    color_mode = "rgb",
    class_mode = "categorical",
    batch_size = BATCH_SIZE,
    shuffle = True
    )


########################### Inspect Loaded Data ##############################
##############################################################################

#plot some sample images for the categories
fun.show_sample_img(
    img_path = TRAIN_DIRECTORY,
    col_names = NAMES,
    save_img = False
    )

#number of observations
train.samples
test.samples

#label equivalence
train.class_indices.keys() == test.class_indices.keys()

#check whether the encoding has worked
train_images, train_labels = train.next()
print("Batch_size: {}, Number of classes: {}".format(train_labels.shape[0], train_labels.shape[1]))


############################## Baseline Model ################################
##############################################################################

#draw baseline model
baseline = keras.Sequential([
    keras.layers.InputLayer(input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
    keras.layers.Conv2D(32, (3, 3), padding="valid", activation="relu"),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(64, (3, 3), activation="relu"),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(128, (3, 3), activation="relu"),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation="softmax"),
    keras.layers.Dense(4)
    ], name = "baseline_model")

baseline.summary()

#define callback
callbacks = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
    ]

#Define optimizer
optimizer = keras.optimizers.SGD(
    learning_rate=0.001,
    nesterov=True
)

#compile model
baseline.compile(
    optimizer = optimizer,
    loss = keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics = ["accuracy"]
    )

#train model
history_baseline = baseline.fit(
    x = train,
    epochs = 30,
    batch_size = BATCH_SIZE,
    callbacks = callbacks,
    validation_data = test,
    verbose=2
    )

#plot loss and accuracy
fun.plot_accuracy(
    history = history_baseline,
    save_location = "..\\plots\\accuracy_baseline.png",
    save = True
    )
fun.plot_loss(
    history = history_baseline,
    save_location = "..\\plots\\loss_baseline.png",
    save = True
    )

#in-sample, out-of-sample performance
train_loss, train_accurary = baseline.evaluate(train, steps = 10)
test_loss, test_accuracy = baseline.evaluate(test, steps = 10)

#confusion matrix & classification report for training and testing data
fun.get_metrics(
    data = train,
    model = baseline
    )
fun.get_metrics(
    data = test,
    model = baseline
    )

#save model
baseline.save("..\\models\\baseline.h5")


############################ Transfer Learning ###############################
##############################################################################

#specify VGG19 Model
vgg19 = VGG19(
    include_top = False,
    weights = "imagenet",
    input_shape = (IMG_WIDTH, IMG_HEIGHT, 3),
    pooling = max
    )

#get model summary
vgg19.summary()

#plot model summary
plot_model(
    model = vgg19, 
    to_file = "..\\plots\\untuned_vgg19.png",
    show_layer_names = True,
    rankdir = 'LR', #horizontal Plot
    expand_nested=True,
    dpi = 300
    )


#use all layers with weights from ImageNet (not necessary since default)
for  layer in vgg19.layers:
    layer.trainable = True

#create Transfer Learning Model that first processes the data with the vgg19 architecture
#and then predicts with newly trained, regularized dense layers.
deepnet = keras.Sequential([
    vgg19,
    keras.layers.Flatten(),
    keras.layers.Dense(1096, activation = "relu", kernel_constraint=max_norm(2.),  bias_constraint=max_norm(3)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1096, activation = "relu", kernel_constraint=max_norm(2.),  bias_constraint=max_norm(3)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(548, activation = "relu", kernel_constraint=max_norm(2.),  bias_constraint=max_norm(3)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(4, activation = "softmax")
    ], name = "deepnet")

#get modelinfo
deepnet.summary()

#plot whole model
plot_model(
    model = deepnet, 
    to_file = "..\\plots\\deepnet.png",
    show_layer_names = True,
    rankdir = "LR", #horizontal Plot
    expand_nested=True,
    dpi = 300
    )

#define callback
callbacks = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
    ]

#define optimizer
optimizer = keras.optimizers.SGD(
    learning_rate=0.001,
    nesterov=True
)

#compile model
deepnet.compile(
    optimizer = optimizer,
    loss = keras.losses.CategoricalCrossentropy(from_logits=False),
    metrics = ["accuracy"]
    )

#fit model
history_deepnet = deepnet.fit(
    x = train,
    epochs = 30,
    batch_size = BATCH_SIZE,
    callbacks = callbacks,
    validation_data = test,
    verbose=2
    )

#save model
deepnet.save("..\\models\\deepnet.h5")


############################ Further Train Model #############################
##############################################################################

# loading the saved model
deepnet = tf.keras.models.load_model("..\\models\\deepnet.h5")

# train model ahead
history_deepnet = deepnet.fit(
    train,
    batch_size = BATCH_SIZE,
    epochs = 1, 
    validation_data = test,
    verbose=2
    )


########################## Evaluate Trained Model ############################
##############################################################################

#in-sample, out-of-sample performance
train_loss, train_accurary = deepnet.evaluate(train, steps = 32)
test_loss, test_accuracy = deepnet.evaluate(test, steps = 32)

#load dataset with propper labels
#ImageDataGenerators for training are used with the shuffle=True argument,
#which is important for propper training (the images must not be passed sorted by
#their labels). However, the model classes (accessed with .classes) are shuffled
#as well and cannot be recovered. This gets problematic when trying to evaluate
#actual labels against predicted labels. Therefore, in order to optain a propper
#confusion matrix, one has to reload train and test datasets with the shuffle 
#argument specified as "False".  
evaluation_train = data_generation_train.flow_from_directory(
    directory = TRAIN_DIRECTORY,
    target_size = (IMG_WIDTH, IMG_HEIGHT),
    color_mode = "rgb",
    class_mode = "categorical",
    batch_size = BATCH_SIZE,
    shuffle = False
    )
evaluation_test = data_generation_test.flow_from_directory(
    directory = TEST_DIRECTORY, 
    target_size = (IMG_WIDTH, IMG_HEIGHT),
    color_mode = "rgb",
    class_mode = "categorical",
    batch_size = BATCH_SIZE,
    shuffle = False
    )

#confusion matrix & classification report
fun.get_metrics(
    data = evaluation_train,
    model = deepnet
    )
fun.get_metrics(
    data = evaluation_test,
    model = deepnet
    )

#plot loss and accuracy
fun.plot_accuracy(
    history = history_deepnet,
    save_location = "..\\plots\\accuracy6567_deepnet.png",
    save = True
    )
fun.plot_loss(
    history = history_deepnet,
    save_location = "..\\plots\\loss55667_deepnet.png",
    save = True
    )