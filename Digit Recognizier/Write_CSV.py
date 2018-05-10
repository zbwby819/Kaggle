# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 15:08:45 2017

@author: Administrator
"""

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from keras.models import load_model


num_classes=10
batch_size = 32
nb_epochs = 80 
kernel_size = 3 
conv_depth = 32  
drop_prob_1 = 0.25 
drop_prob_2 = 0.5 
l2_lambda = 0.02 
data_augrate=1
data_augmentation = False


data = pd.read_csv('test.csv')
images = data.iloc[:,:].values
images = images.astype(np.float)
images = np.multiply(images, 1.0 / 255.0)
image_size = images.shape[1]
image_width = image_height = np.ceil(np.sqrt(image_size)).astype(np.uint8)


X_test=images[:,:]
X_test = X_test.reshape(X_test.shape[0],image_width,image_height,1)

model = load_model('DigitRC.h5')

preds = model.predict_classes(X_test, verbose=0)

def write_preds(preds, fname):
    pd.DataFrame({"ImageId": list(range(1,len(preds)+1)), "Label": preds}).to_csv(fname, index=False, header=True)

write_preds(preds, "keras.csv")