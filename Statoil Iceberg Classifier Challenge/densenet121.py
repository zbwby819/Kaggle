# -*- coding: utf-8 -*-

from keras.layers import Input, merge, ZeroPadding2D
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Flatten,  Average
from keras.layers.merge import Concatenate
from keras import initializers
from keras import regularizers
import keras.optimizers 
from keras.regularizers import l2 
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, Callback, EarlyStopping
import keras.backend as K

from custom_layers.scale_layer import Scale

#from load_cifar10 import load_cifar10_data

import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
#from os.path import join as opj
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#import pylab
import os
import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


#os.environ["PATH"] += os.pathsep + 'D:/Kallge/IceBerg/bin/'
#from matplotlib import pyplot

reduce_lr = ReduceLROnPlateau(monitor='loss', 
                              patience=2, 
                              verbose=1, 
                              factor=0.5, 
                              min_lr=0.000001)

best_weights_filepath = 'kaggle/IceBerg/desmodel1.h5py'
#earlyStopping=EarlyStopping(monitor='val_loss', patience=30, verbose=1, mode='min')

saveBestModel = ModelCheckpoint(best_weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')




split=0.2
batch_size = 16
nb_classes = 1
nb_epoch =15
l2_lambda=0.00
data_augmentation=False
rate=5
seed=7
rows, cols = 224 , 224
channels = 3

train = pd.read_json("kaggle/IceBerg/data/train.json")
test = pd.read_json("kaggle/IceBerg/data/test.json")

xb1=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_1"]])
xb2=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_2"]])
train_target=train['is_iceberg']
X_train3 = xb1[:,20:60,20:60]


resized1 = []
resized2 = []
resized3 = []
for i in range(xb1.shape[0]):    
    resize1 = cv2.resize(xb1[i], (rows, cols))
    resized1.append(resize1)
    resize2 = cv2.resize(xb2[i], (rows, cols))
    resized2.append(resize2)
    resize3 = cv2.resize(X_train3[i], (rows, cols))
    resized3.append(resize3)

X_train1 = np.array(resized1, dtype=np.float32)
X_train2 = np.array(resized2, dtype=np.float32)
X_train3 = np.array(resized3, dtype=np.float32)
#X_train1 /= 255
#X_train2 /= 255


X_train = np.concatenate([X_train1[:, :, :, np.newaxis], X_train2[:, :, :, np.newaxis],X_train3[:, :, :, np.newaxis]], axis=-1)    
X_train, X_valid, y_train, y_valid = train_test_split(X_train, train_target, random_state=seed, train_size=1-split)



def densenet121_model(img_rows, img_cols, color_type=1, nb_dense_block=4, growth_rate=32, nb_filter=64, reduction=0.5, dropout_rate=0., weight_decay=1e-4, num_classes=None):

    eps = 1.1e-5

    # compute compression factor
    compression = 1.0 - reduction

    # Handle Dimension Ordering for different backends
    global concat_axis
    if K.image_dim_ordering() == 'tf':
      concat_axis = 3
      img_input = Input(shape=(img_rows, img_cols, color_type), name='data')
    else:
      concat_axis = 1
      img_input = Input(shape=(color_type, img_rows, img_cols), name='data')

    # From architecture for ImageNet (Table 1 in the paper)
    nb_filter = 64
    nb_layers = [6,12,24,16] # For DenseNet-121

    # Initial convolution
    x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)
    x = Conv2D(nb_filter, kernel_size=(7, 7), strides=(2, 2), name='conv1', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv1_bn')(x)
    x = Scale(axis=concat_axis, name='conv1_scale')(x)
    x = Activation('relu', name='relu1')(x)
    x = ZeroPadding2D((1, 1), name='pool1_zeropadding')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)

    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        stage = block_idx+2
        x, nb_filter = dense_block(x, stage, nb_layers[block_idx], nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)

        # Add transition_block
        x = transition_block(x, stage, nb_filter, compression=compression, dropout_rate=dropout_rate, weight_decay=weight_decay)
        nb_filter = int(nb_filter * compression)

    final_stage = stage + 1
    x, nb_filter = dense_block(x, final_stage, nb_layers[-1], nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)

    x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv'+str(final_stage)+'_blk_bn')(x)
    x = Scale(axis=concat_axis, name='conv'+str(final_stage)+'_blk_scale')(x)
    x = Activation('relu', name='relu'+str(final_stage)+'_blk')(x)

    x_fc = GlobalAveragePooling2D(name='pool'+str(final_stage))(x)
    x_fc = Dense(1000, name='fc6')(x_fc)
    x_fc = Activation('softmax', name='prob')(x_fc)

    model = Model(img_input, x_fc, name='densenet')

    if K.image_dim_ordering() == 'th':
      # Use pre-trained weights for Theano backend
      weights_path = 'pre_train/imagenet_models/densenet121_weights_th.h5'
    else:
      # Use pre-trained weights for Tensorflow backend
      weights_path = 'pre_train/imagenet_models/densenet121_weights_tf.h5'

    model.load_weights(weights_path, by_name=True)

    # Truncate and replace softmax layer for transfer learning
    # Cannot use model.layers.pop() since model is not of Sequential() type
    # The method below works since pre-trained weights are stored in layers but not in the model
    x_newfc = GlobalAveragePooling2D(name='pool'+str(final_stage))(x)
    x_newfc = Dropout(0.5)(x_newfc)
    x_newfc = Dense(num_classes, name='fc6',kernel_regularizer=regularizers.l2(l2_lambda),kernel_initializer='he_uniform')(x_newfc)
    x_newfc = Activation('sigmoid', name='prob')(x_newfc)

    model = Model(img_input, x_newfc)

    nadam1=keras.optimizers.Nadam(lr=0.0003, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.0)
    #sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=nadam1, loss='binary_crossentropy', metrics=['accuracy'])

    return model


def conv_block(x, stage, branch, nb_filter, dropout_rate=None, weight_decay=1e-4):
    '''Apply BatchNorm, Relu, bottleneck 1x1 Conv2D, 3x3 Conv2D, and option dropout
        # Arguments
            x: input tensor 
            stage: index for dense block
            branch: layer index within each dense block
            nb_filter: number of filters
            dropout_rate: dropout rate
            weight_decay: weight decay factor
    '''
    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_' + str(branch)
    relu_name_base = 'relu' + str(stage) + '_' + str(branch)

    # 1x1 Convolution (Bottleneck layer)
    inter_channel = nb_filter * 4  
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_x1_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base+'_x1_scale')(x)
    x = Activation('relu', name=relu_name_base+'_x1')(x)
    x = Conv2D(inter_channel, kernel_size=(1, 1), strides=(1, 1), name=conv_name_base+'_x1', use_bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    # 3x3 Convolution
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_x2_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base+'_x2_scale')(x)
    x = Activation('relu', name=relu_name_base+'_x2')(x)
    x = ZeroPadding2D((1, 1), name=conv_name_base+'_x2_zeropadding')(x)
    x = Conv2D(nb_filter, kernel_size=(3, 3), strides=(1, 1), name=conv_name_base+'_x2', use_bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def transition_block(x, stage, nb_filter, compression=1.0, dropout_rate=None, weight_decay=1E-4):
    ''' Apply BatchNorm, 1x1 Convolution, averagePooling, optional compression, dropout 
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_filter: number of filters
            compression: calculated as 1 - reduction. Reduces the number of feature maps in the transition block.
            dropout_rate: dropout rate
            weight_decay: weight decay factor
    '''

    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_blk'
    relu_name_base = 'relu' + str(stage) + '_blk'
    pool_name_base = 'pool' + str(stage) 

    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base+'_scale')(x)
    x = Activation('relu', name=relu_name_base)(x)
    x = Conv2D(int(nb_filter * compression), kernel_size=(1, 1), strides=(1, 1), name=conv_name_base, use_bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    x = AveragePooling2D((2, 2), strides=(2, 2), name=pool_name_base)(x)

    return x


def dense_block(x, stage, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1e-4, grow_nb_filters=True):
    ''' Build a dense_block where the output of each conv_block is fed to subsequent ones
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_layers: the number of layers of conv_block to append to the model.
            nb_filter: number of filters
            growth_rate: growth rate
            dropout_rate: dropout rate
            weight_decay: weight decay factor
            grow_nb_filters: flag to decide to allow number of filters to grow
    '''

    eps = 1.1e-5
    concat_feat = x

    for i in range(nb_layers):
        branch = i+1
        x = conv_block(concat_feat, stage, branch, growth_rate, dropout_rate, weight_decay)
        concat_feat = merge([concat_feat, x], mode='concat', concat_axis=concat_axis, name='concat_'+str(stage)+'_'+str(branch))

        if grow_nb_filters:
            nb_filter += growth_rate

    return concat_feat, nb_filter

if __name__ == '__main__':


    # Load our model
    model = densenet121_model(img_rows=rows, img_cols=cols, color_type=channels, num_classes=nb_classes)


    if not data_augmentation:
        print('Not using data augmentation.')
        model.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=nb_epoch,
                  validation_data=(X_valid, y_valid),
                  #shuffle=True
                  callbacks=[reduce_lr]
       	      )
    else:
        print('Using real-time data augmentation.')

        # this will do preprocessing and realtime data augmentation
        datagen = ImageDataGenerator(
                featurewise_center=False,  # set input mean to 0 over the dataset
                samplewise_center=False,  # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,  # divide each input by its std
                zca_whitening=False,  # apply ZCA whitening
                shear_range=0.,
                zoom_range=0.2,
                rotation_range=30,  # randomly rotate images in the range (degrees, 0 to 180)
                width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
                height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
                horizontal_flip=True,  # randomly flip images
                vertical_flip=True)  # randomly flip images

        # compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied)
        datagen.fit(X_train)

        # fit the model on the batches generated by datagen.flow()
        model.fit_generator(datagen.flow(X_train, y_train,
                                         batch_size=batch_size),
                                         steps_per_epoch=X_train.shape[0]*rate/batch_size,
                                         epochs=nb_epoch,
                                         validation_data=(X_valid, y_valid),
                                         callbacks=[saveBestModel, reduce_lr]
			                 )

    
    X_band_test_1=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_1"]])
    X_band_test_2=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_2"]])
    X_band_test_3 = X_band_test_1[:,20:60,20:60]

    resized6 = []
    resized4 = []
    resized5 = []
    for i in range(X_band_test_2.shape[0]):    
        resize6 = cv2.resize(X_band_test_1[i], (rows, cols))
        resized6.append(resize6)
        resize4 = cv2.resize(X_band_test_2[i], (rows, cols))
        resized4.append(resize4)
        resize5 = cv2.resize(X_band_test_3[i], (rows, cols))
        resized5.append(resize5)


    X_test1 = np.array(resized6, dtype=np.float32)
    #X_test1 /= 255
    X_test2 = np.array(resized4, dtype=np.float32)
    #X_test2 /= 255
    X_test3 = np.array(resized5, dtype=np.float32)
    #X_test3 /= 255
    X_test = np.concatenate([X_test1[:, :, :, np.newaxis], X_test2[:, :, :, np.newaxis],
                             ((X_test3+X_test3)/2)[:, :, :, np.newaxis]], axis=-1)
    
    
    model.load_weights(filepath=best_weights_filepath)
    score = model.evaluate(X_valid, y_valid, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    predicted_test=model.predict(X_test)

    submission = pd.DataFrame({'id': test["id"], 'is_iceberg': predicted_test.reshape((predicted_test.shape[0]))})
    submission.to_csv('kaggle/IceBerg/sub1.csv', index=False)

