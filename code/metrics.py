# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 20:12:45 2021

@author: Kilian
"""
################################### Input ####################################
##############################################################################
#Libraries
import os
import pandas as pd
import matplotlib.pyplot as plt

#setwd
PATH = "D:\\Bibliotheken\\Dokumente\\GitHub\\MRIAlzheimersImagePrediction\\code"
#PATH = "C:\\Users\\kilia\\Documents\\GitHub\\MRIAlzheimersImagePrediction\\code"
os.chdir(PATH)

#read dataframe
X = pd.read_csv("..\\data\\deepnet_metrics.csv",header=None)


################################ Wrangling ###################################
##############################################################################

#rename column to col
X = X.rename(columns = {0 : "col"})

#split the column by "-"
X = X["col"].str.split("-", expand = True)

#drop the first row as it is not needed
X = X.drop([0], axis=1)

#rename rows
names = ["secs_of_training", "loss_training", "accuracy_training", "loss_validation", "accuracy_validation"]
X.columns = names

#clean up cells and convert them to integers and floats
X["secs_of_training"] = X["secs_of_training"].str.replace("s ", "").astype(int)
type(X["secs_of_training"][3])

X["loss_training"] = X["loss_training"].str.replace(" loss: ", "").astype(float)
type(X["loss_training"][3])

X["accuracy_training"] = X["accuracy_training"].str.replace(" accuracy: ", "").astype(float)
type(X["accuracy_training"][3])

X["loss_validation"] = X["loss_validation"].str.replace(" val_loss: ", "").astype(float)
type(X["loss_validation"][3])

X["accuracy_validation"] = X["accuracy_validation"].str.replace(" val_accuracy: ", "").astype(float)
type(X["accuracy_validation"][3])

#export cleaned dataset to excel
X.to_excel("..\\data\\deepnet_metrics_clean.xlsx")


############################### Visualization ################################
##############################################################################

#plot loss
plt.plot(X["loss_training"])
plt.plot(X["loss_validation"])
plt.title("deepnet loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["training", "validation"], loc="upper right")
plt.savefig(fname = "..\\plots\\deepnet_loss.png", dpi=300)

#plot accuracy
plt.plot(X["accuracy_training"])
plt.plot(X["accuracy_validation"])
plt.title("deepnet accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(["training", "validation"], loc="upper left")
plt.savefig(fname = "..\\plots\\deepnet_accuracy.png", dpi=300)