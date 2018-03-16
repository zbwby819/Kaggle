# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 16:36:14 2017

@author: Administrator
"""

import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from os.path import join as opj
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pylab
import os
import cv2

from matplotlib import pyplot
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, Activation, Average
from keras.layers import GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Model
from keras import initializers
import keras.optimizers 
from keras import regularizers
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, Callback, EarlyStopping



learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.000001)


split=0.2
batch_size = 32
#nb_classes = 1
nb_epoch =100
l2_lambda=0.005
data_augmentation=False
data_augrate=1
seed=2

train = pd.read_json("data/train.json")
test = pd.read_json("data/test.json")

xb1=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_1"]])
xb2=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_2"]])
X_train = np.concatenate([xb1[:, :, :, np.newaxis], xb2[:, :, :, np.newaxis],((xb1+xb2)/2)[:, :, :, np.newaxis]], axis=-1)
train_target=train['is_iceberg']


X_train, X_valid, y_train, y_valid = train_test_split(X_train, train_target, random_state=seed, train_size=1-split)



rows, cols = xb1.shape[1] , xb1.shape[2]
channels = 3

input_shape = X_train[0,:,:,:].shape
model_input = Input(shape=input_shape)
#model_input = BatchNormalization()(inp)

def globalpool(model_input):
    x = Conv2D(96, kernel_size=(3, 3), activation='relu', padding = 'same')(model_input)
    x = Conv2D(96, (3, 3), activation='relu', padding = 'same')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides = 2)(x)
    x = Conv2D(192, (3, 3), activation='relu', padding = 'same')(x)
    x = Conv2D(192, (3, 3), activation='relu', padding = 'same')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides = 2)(x)
    x = Conv2D(384, (3, 3), activation='relu', padding = 'same')(x)
    x = Conv2D(384, (3, 3), activation='relu', padding = 'same')(x)
    x = Conv2D(1, (1, 1))(x)
    x = GlobalAveragePooling2D()(x)
    x = Activation(activation='sigmoid')(x)
    model = Model(model_input, x, name='globalpool')
    return model


def all_cnn(model_input):
    x = Conv2D(96, kernel_size=(3, 3), activation='relu', padding = 'same')(model_input)
    x = Conv2D(96, (3, 3), activation='relu', padding = 'same')(x)
    x = Conv2D(96, (3, 3), activation='relu', padding = 'same', strides = 2)(x)
    x = Conv2D(192, (3, 3), activation='relu', padding = 'same')(x)
    x = Conv2D(192, (3, 3), activation='relu', padding = 'same')(x)
    x = Conv2D(192, (3, 3), activation='relu', padding = 'same', strides = 2)(x)
    x = Conv2D(384, (3, 3), activation='relu', padding = 'same')(x)
    x = Conv2D(384, (3, 3), activation='relu', padding = 'same')(x)
    x = Conv2D(1, (1, 1))(x)
    x = GlobalAveragePooling2D()(x)
    x = Activation(activation='sigmoid')(x)
    model = Model(model_input, x, name='all_cnn')
    return model


def nin_cnn(model_input):
    #mlpconv block 1
    x = Conv2D(96, (3, 3), activation='relu',padding='same')(model_input)
    x = Conv2D(96, (1, 1), activation='relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides = 2)(x)
    x = Dropout(0.5)(x)
    #mlpconv block2
    x = Conv2D(192, (3, 3), activation='relu',padding='same')(x)
    x = Conv2D(192, (1, 1), activation='relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides = 2)(x)
    x = Dropout(0.5)(x)
    #mlpconv block3
    x = Conv2D(384, (3, 3), activation='relu',padding='same')(x)
    x = Conv2D(384, (1, 1), activation='relu')(x)
    x = Conv2D(1, (1, 1))(x)
    x = GlobalAveragePooling2D()(x)
    x = Activation(activation='sigmoid')(x)
    model = Model(model_input, x, name='nin_cnn')
    return model

gpmodel = globalpool(model_input)
acmodel = all_cnn(model_input)
ninmodel = nin_cnn(model_input)
enmodels = [gpmodel, acmodel, ninmodel]

def ensemble(models, model_input):
    outputs = [model.outputs[0] for model in models]
    y = Average()(outputs)
    model = Model(model_input, y, name='enmodels')
    return model

model = ensemble(enmodels, model_input)

'''
def getModel():
    model=Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same',activation='relu',input_shape=(rows, cols, channels)))
    model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same',activation='relu'))
    #model.add(Conv2D(64, kernel_size=(3, 3), strides=(2, 2), padding='valid',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(128,kernel_size=(3, 3), strides=(1, 1), padding='same',activation='relu'))
    model.add(Conv2D(128,kernel_size=(3, 3), strides=(1, 1), padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(256,kernel_size=(3, 3), strides=(1, 1), padding='same',activation='relu'))
    model.add(Conv2D(256,kernel_size=(3, 3), strides=(1, 1), padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(512,kernel_size=(3, 3), strides=(1, 1), padding='same',activation='relu'))
    model.add(Conv2D(512,kernel_size=(3, 3), strides=(1, 1), padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(512,kernel_regularizer=regularizers.l2(l2_lambda)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))


    return model
'''

nadam1=keras.optimizers.Nadam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
model.compile(loss='binary_crossentropy',
                  optimizer=nadam1,
                  metrics=['accuracy'])

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=nb_epoch,
              validation_data=(X_valid, y_valid),
              shuffle=True)
else:
    print('Using real-time data augmentation.')

    # this will do preprocessing and realtime data augmentation
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(X_train)

    # fit the model on the batches generated by datagen.flow()
    model.fit_generator(datagen.flow(X_train, y_train),
                        batch_size=batch_size,
                        steps_per_epoch=X_train.shape[0]/(batch_size/data_augrate),
                        epochs=nb_epoch,
                        validation_data=(X_valid, y_valid))


X_band_test_1=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_1"]])
X_band_test_2=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_2"]])
X_test = np.concatenate([X_band_test_1[:, :, :, np.newaxis], X_band_test_2[:, :, :, np.newaxis],
                       ((X_band_test_1+X_band_test_2)/2)[:, :, :, np.newaxis]], axis=-1)
predicted_test=model.predict_proba(X_test)


submission = pd.DataFrame()
submission['id']=test['id']
submission['is_iceberg']=predicted_test.reshape((predicted_test.shape[0]))
submission.to_csv('sub.csv', index=False)
