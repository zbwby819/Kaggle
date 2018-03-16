# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 16:36:14 2017

@author: Administrator
"""

import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from os.path import join as opj
#import matplotlib.pyplot as plt
import os

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, Activation
from keras.layers import GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Model
from keras import initializers
from sklearn.model_selection import StratifiedKFold
import keras.optimizers 
from keras import regularizers
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, Callback, EarlyStopping
from keras.applications.resnet50 import ResNet50

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


reduce_lr = ReduceLROnPlateau(monitor='loss', 
                                            patience=9, 
                                            verbose=1, 
                                            factor=0.5, 
					    mode='min',
					    #cooldown=0,
                                            min_lr=0.000001)


best_weights_filepath = 'kaggle/IceBerg/model1.h5py'
earlyStopping=EarlyStopping(monitor='val_loss', patience=60, verbose=1, mode='min')
saveBestModel = ModelCheckpoint(best_weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')


split=0.2
batch_size = 64
#nb_classes = 1
nb_epoch =300
l2_lambda=0.00
data_augmentation=True
rate=3
seed=3
pre_threshold=0.01
K=10



train = pd.read_json("kaggle/IceBerg/data/train.json")
test = pd.read_json("kaggle/IceBerg/data/test.json")

xb1=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_1"]])
xb2=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_2"]])
#X_train = np.concatenate([xb1[:, :, :, np.newaxis], xb2[:, :, :, np.newaxis],((xb1+xb2)/2)[:, :, :, np.newaxis]], axis=-1)
train_target=train['is_iceberg']
xb3 = xb1[:,20:60,20:60]
xb4 = xb2[:,20:60,20:60]
X_train = np.concatenate([xb1[:, :, :, np.newaxis], xb2[:, :, :, np.newaxis],((xb1+xb2)/2)[:, :, :, np.newaxis]], axis=-1)

#X_train, X_valid, y_train, y_valid = train_test_split(X_train, train_target, random_state=seed, test_size = split)
folds = list(StratifiedKFold(n_splits=K, shuffle=True, random_state=16).split(X_train, train_target))

rows, cols = xb3.shape[1] , xb4.shape[2]
channels = 3


X_band_test_1=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_1"]])
X_band_test_2=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_2"]])
xt3 = X_band_test_1[:,20:60,20:60]
xt4 = X_band_test_2[:,20:60,20:60]
X_test = np.concatenate([xt3[:, :, :, np.newaxis], xt4[:, :, :, np.newaxis],((xt3+xt4)/2)[:, :, :, np.newaxis]], axis=-1)
#X_test = np.concatenate([X_band_test_1[:, :, :, np.newaxis], X_band_test_2[:, :, :, np.newaxis],
#                       ((X_band_test_1+X_band_test_2)/2)[:, :, :, np.newaxis]], axis=-1)


def getModel():
    model=Sequential()
    model.add(Conv2D(128, kernel_size=(5, 5), strides=(1, 1), padding='same',activation='relu',input_shape=(rows, cols, channels)))
    model.add(Conv2D(128, kernel_size=(1, 1), strides=(1, 1), padding='same',activation='relu'))
    #model.add(Conv2D(64, kernel_size=(3, 3), strides=(2, 2), padding='valid',activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(Conv2D(192,kernel_size=(3, 3), strides=(1, 1), padding='same',activation='relu'))
    model.add(Conv2D(192,kernel_size=(1, 1), strides=(1, 1), padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(Conv2D(256,kernel_size=(3, 3), strides=(1, 1), padding='same',activation='relu'))
    model.add(Conv2D(256,kernel_size=(1, 1), strides=(1, 1), padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    #model.add(Conv2D(512,kernel_size=(3, 3), strides=(1, 1), padding='same',activation='relu'))
    #model.add(Conv2D(512,kernel_size=(1, 1), strides=(1, 1), padding='same',activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    #model.add(Conv2D(512,kernel_size=(3, 3), strides=(1, 1), padding='same',activation='relu'))
    #model.add(Conv2D(512,kernel_size=(3, 3), strides=(1, 1), padding='same',activation='relu'))
    #model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(2048,kernel_regularizer=regularizers.l2(l2_lambda)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2048,kernel_regularizer=regularizers.l2(l2_lambda)))
    #kernel_initializer='he_uniform'
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    nadam1=keras.optimizers.Nadam(lr=0.00005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
    model.compile(loss='binary_crossentropy',
                  optimizer=nadam1,
                  metrics=['accuracy'])
    return model


y_test_pred_log = 0
y_train_pred_log=0

for j, (train_idx, test_idx) in enumerate(folds):
    print('\n===================FOLD=',j)
    X_train_cv = X_train[train_idx]
    y_train_cv = train_target[train_idx]
    X_holdout = X_train[test_idx]
    Y_holdout= train_target[test_idx]
   
    model=getModel()
    if not data_augmentation:
    	 print('Not using data augmentation.')
   	 model.fit(X_train_cv, y_train_cv,
    	          batch_size=batch_size,
    	          epochs=nb_epoch,
     	          validation_data=(X_holdout, Y_holdout),
		  callbacks=[earlyStopping,saveBestModel,reduce_lr]
 	          )
    else:
   	 print('Using real-time data augmentation.')

  	 #this will do preprocessing and realtime data augmentation
  	 datagen = ImageDataGenerator(
   	  	featurewise_center=False,  # set input mean to 0 over the dataset
 	    	samplewise_center=False,  # set each sample mean to 0
  	        featurewise_std_normalization=False,  # divide inputs by std of the dataset
   	     	samplewise_std_normalization=False,  # divide each input by its std
        	zca_whitening=False,  # apply ZCA whitening
        	zoom_range=0.2,
		rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        	width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        	height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        	horizontal_flip=False,  # randomly flip images
        	vertical_flip=False)  # randomly flip images

    	# compute quantities required for featurewise normalization
    	# (std, mean, and principal components if ZCA whitening is applied)
   	 datagen.fit(X_train_cv)

    	# fit the model on the batches generated by datagen.flow()
    	 model.fit_generator(datagen.flow(X_train_cv, y_train_cv,
        	                batch_size=batch_size),
                	        steps_per_epoch=X_train.shape[0]*rate/batch_size,
                        	epochs=nb_epoch,
                      		validation_data=(X_holdout, Y_holdout),
				callbacks=[earlyStopping,saveBestModel,reduce_lr]
				#shuffle=False, 
				)
    
    model.load_weights(filepath=best_weights_filepath)
    score=model.evaluate(X_holdout,Y_holdout,verbose=1)
    print('\n===================FOLD=',j)
    print('Valid loss:', score[0])
    print('Valid accuracy:', score[1])

    temp_test=model.predict_proba(X_test)
    y_test_pred_log+=temp_test.reshape(temp_test.shape[0])


predicted_test=y_test_pred_log/K

#predicted_test=model.predict_proba(X_test)
#predicted_test=np.clip(predicted_test,pre_threshold,1-pre_threshold)


submission = pd.DataFrame()
submission['id']=test['id']
submission['is_iceberg']=predicted_test.reshape((predicted_test.shape[0]))
submission.to_csv('kaggle/IceBerg/sub.csv', index=False)
