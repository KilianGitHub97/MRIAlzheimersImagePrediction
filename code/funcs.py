# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 11:14:39 2021

@author: kilia
"""
# imports
import os
import numpy as np
import random
import shutil
import matplotlib.pyplot as plt
import matplotlib.image as im

######################## count observations in directory #####################
def count_obs(folder_path):
    tot_files = 0
    for unneeded1, unneeded2, files in os.walk(folder_path):
        for file in files:
            tot_files += 1
    return tot_files

################# calculate batch size without remainer #####################
def calc_batch_size_no_remainer(observations, max_batch_size, min_batch_size):
    
    #adjust max batch size if necessary
    if max_batch_size >= observations:
            max_batch_size = observations - 1
    
    #check for whether the modulus is 0 
    for denominator in range(max_batch_size, min_batch_size, -1):
        modulus = observations % denominator
        if modulus == 0:
            if denominator == 1:
                print("There is no denominator in your specified range (other than "
                      "1) that can divide your number of observations without a remainer")
            else:
                print("The first number to divide your observations without "
                      "a remainer is {}.".format(denominator))
            break
##################### get class labels as a list ##########################     
def get_class_names(folder_path):
    classnames = []
    for unneeded1, folders, unneeded2 in os.walk(folder_path):
        for folder in folders:
            classnames.append(folder)
    return classnames

################## check the ballance of the data folder ##################
def balance_check(folder_path, class_names, number_of_obs, normalize=False):
    obs_per_class = []
    for folder_num in range(4):
        new_path = folder_path+"\\"+class_names[folder_num]
        file_count = 0
        for unneeded, unneeded2, files in os.walk(new_path):
            for file in files:
                file_count += 1
        obs_per_class.append(file_count)
    if normalize == True:
        norm_obs_per_class = []
        for value in range(4):
            norm_obs_per_class.append(obs_per_class[value] / number_of_obs)
        obs_per_class = norm_obs_per_class
    output = dict(zip(class_names, obs_per_class))
    return output


########################## draw sample images #############################
def show_sample_img(img_path, col_names):
    
    # empty list to fill with images (nparrays)
    images = []
    
    # load a random image from the given path and add it to the images-list
    for folder in range(4):
        rand_img_dir = img_path+"\\"+col_names[folder]
        rand_img = random.choice(os.listdir(rand_img_dir))
        rand_img_open = rand_img_dir+"\\"+rand_img
        image = im.imread(rand_img_open)
        images.append(image)
            
    # 2x2 Plot with 4 pictures, each from an individual category
    fig, ((pic1, pic2), (pic3, pic4)) = plt.subplots(nrows=2, ncols=2)
    fig.suptitle('Sample images of the four classes')
    pic1.set_title(col_names[0])
    pic2.set_title(col_names[1])
    pic3.set_title(col_names[2])
    pic4.set_title(col_names[3])
    pic1.imshow(images[0], cmap="gray")
    pic2.imshow(images[1], cmap="gray")
    pic3.imshow(images[2], cmap="gray")
    pic4.imshow(images[3], cmap="gray")
    fig.tight_layout()
    
###################### make test folder ##########################
def make_test_folder(test_directory, train_directory, classes, total_number_of_observations, percentage_test_directory):
    
    #create test folder
    os.makedirs(test_directory)
    
    #get observations per class
    labels = balance_check(
        folder_path = train_directory,
        class_names = classes,
        number_of_obs = total_number_of_observations,
        normalize=False
        )
    
    for class_name in classes:
        #create directory strings
        test_path = test_directory+"\\"+class_name
        train_path = train_directory+"\\"+class_name
        
        #get the number of images to be shifted into the test folder
        total_observations_per_class = labels.get(class_name)
        twenty_percent_of_total_obs = round(((total_observations_per_class / 100)*percentage_test_directory), 0)
        
        #new folder with label
        os.makedirs(test_path)
                
        #select random images from train folder and move it to test folder
        num_files = 0
        while num_files <= twenty_percent_of_total_obs:
            rand_image = random.choice(os.listdir(train_path))
            shutil.move(train_path+"\\"+rand_image, test_path)
            num_files += 1



        