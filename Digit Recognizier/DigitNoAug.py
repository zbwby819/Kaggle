# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 09:29:05 2017

@author: Administrator
"""


import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt


import keras.optimizers 
from keras.utils import np_utils 
from keras.layers import Dense, Flatten, Convolution2D, MaxPooling2D, Dropout, Activation
from keras.preprocessing.image import ImageDataGenerator 
from keras.models import Sequential
from keras.regularizers import l2 
from keras.layers.normalization import BatchNormalization


num_classes=10
batch_size = 32
nb_epochs = 2 
kernel_size = 3 
conv_depth = 32  
drop_prob_1 = 0.25 
drop_prob_2 = 0.5 
l2_lambda = 0.00 
data_augrate=1
data_augmentation = False



data = pd.read_csv('train.csv')
images = data.iloc[:,1:].values
images = images.astype(np.float)
images = np.multiply(images, 1.0 / 255.0)
image_size = images.shape[1]
image_width = image_height = np.ceil(np.sqrt(image_size)).astype(np.uint8)


labels = data.iloc[:,0].values
labels = labels.astype(np.float)



data1 = pd.read_csv('test.csv')
images1 = data1.iloc[:,:].values
images1 = images1.astype(np.float)
images1 = np.multiply(images1, 1.0 / 255.0)


X_train = images[:,:]
X_test=images1[:,:]
Y_train = np_utils.to_categorical(labels[:], num_classes) 
#Y_test = np_utils.to_categorical(labels[30001:], num_classes) 

X_train = X_train.reshape(X_train.shape[0],image_width,image_height,1)
X_test = X_test.reshape(X_test.shape[0],image_width,image_height,1)




'''
choose=333
one_image = images[choose].reshape(image_width,image_height)
label=labels[choose]
plt.imshow(one_image)
print(label)
'''
model = Sequential()

model.add(Convolution2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same',activation='relu',input_shape=X_train.shape[1:]))
model.add(Convolution2D(64, kernel_size=(1, 1), strides=(1, 1), padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Convolution2D(128,kernel_size=(3, 3), strides=(1, 1), padding='same',activation='relu'))
model.add(Convolution2D(128,kernel_size=(1, 1), strides=(1, 1), padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))



model.add(Flatten())
model.add(Dense(48,kernel_regularizer=l2(l2_lambda)))
model.add(Activation('relu'))
#model.add(Dropout(0.25))
model.add(Dense(48,kernel_regularizer=l2(l2_lambda)))
model.add(Activation('relu'))
#model.add(Dropout(0.25))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# let's train the model using SGD + momentum (how original).
nadam1=keras.optimizers.Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
model.compile(loss='categorical_crossentropy',
              optimizer=nadam1,
              metrics=['accuracy'])

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=nb_epochs,
              #validation_data=(X_test, Y_test),
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
        rotation_range=30,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=5/28,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=5/28,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(X_train)

    # fit the model on the batches generated by datagen.flow()
    model.fit_generator(datagen.flow(X_train, Y_train,
                        batch_size=batch_size),
                        steps_per_epoch=X_train.shape[0]/(batch_size/data_augrate),
                        epochs=nb_epochs,
                        #validation_data=(X_test, Y_test)
                        )
    
    
preds = model.predict_classes(X_test, verbose=0)







model.save('Digit.h5')

def write_preds(preds, fname):
    pd.DataFrame({"ImageId": list(range(1,len(preds)+1)), "Label": preds}).to_csv(fname, index=False, header=True)

write_preds(preds, "keras.csv")