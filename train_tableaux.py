# -*- coding: utf-8 -*-
"""
@author: Louis
"""

import keras
from keras.models import Model
from keras.layers import Flatten, Reshape, Input, Cropping2D, Concatenate
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Dense
from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing.image import img_to_array, array_to_img
from keras.utils import np_utils

import numpy as np

from PIL import Image, ImageChops

import matplotlib.pyplot as plt

from game import DICT_TABLEAUX, LIST_TABLEAUX, N_TABLEAUX

N_IMG_PAR_TABLEAUX = 55

IMG_H = 600
IMG_W = 800

def toOneHot(strTab, vocab):
    encoded = np_utils.to_categorical(strTab, num_classes=len(vocab))
    
    return encoded

def genTableaux():
    while True:
        #Choisi le type de tableau
        key = np.random.choice(LIST_TABLEAUX)
        
        #Recupere le tableau
        img = Image.open("gta_rsrc/tableaux/{0}_{1}.png".format(key, np.random.randint(0, N_IMG_PAR_TABLEAUX)))

        #Genere un bruit uniform
        alp = np.random.randint(0, 128)
        col = np.random.randint(0, 255)
        rnd = np.random.random_sample(IMG_H*IMG_W)
        rnd = np.multiply(rnd, alp)
        rnd = np.ndarray.astype(rnd, dtype=int)
        rnd = np.reshape(rnd, (1, IMG_H*IMG_W))
        rnd = np.transpose(rnd)
        rnd = np.insert(rnd, 0, col, axis=-1)
        rnd = np.insert(rnd, 0, col, axis=-1)
        rnd = np.insert(rnd, 0, col, axis=-1)
        rnd = np.reshape(rnd, (IMG_H, IMG_W, 4))
        ind = array_to_img(rnd)
        img.paste(ind, mask=ind)
        
        #Retourn le resultat
        yield img, key

def genDatasetTableaux(N=1):
    gen = genTableaux()

    while True:
        n_train = N
        x_train = []
        y_train = []
        
        for i in range(n_train):
            x, y = next(gen)
            x_train.append(img_to_array(x))
            y_train.append(toOneHot(y, LIST_TABLEAUX))

        yield np.array(x_train), np.array(y_train)

def createTableauxModel():
    inputs = Input(shape=(IMG_H, IMG_W, 3))

    #Réseau CNN
    cnn_a1 = Conv2D(32, kernel_size=3, strides=1, padding='valid', activation=LeakyReLU(), data_format='channels_last')(inputs)
    cnn_a2 = Conv2D(32, kernel_size=3, strides=1, padding='valid', activation=LeakyReLU(), data_format='channels_last')(cnn_a1)
    cnn_a3 = BatchNormalization()(cnn_a2)
    cnn_a4 = MaxPooling2D(pool_size=2)(cnn_a3)
    cnn_a5 = Conv2D(16, kernel_size=3, strides=2, padding='valid', activation=LeakyReLU(), data_format='channels_last')(cnn_a4)
    cnn_a6 = Conv2D(16, kernel_size=3, strides=2, padding='valid', activation=LeakyReLU(), data_format='channels_last')(cnn_a5)
    cnn_a7 = BatchNormalization()(cnn_a6)
    cnn_a8 = MaxPooling2D(pool_size=2)(cnn_a7)
    cnn_a9 = Conv2D( 8, kernel_size=3, strides=4, padding='valid', activation=LeakyReLU(), data_format='channels_last')(cnn_a8)
    cnn_aA = Conv2D( 8, kernel_size=3, strides=4, padding='valid', activation=LeakyReLU(), data_format='channels_last')(cnn_a9)
    cnn_aB = BatchNormalization()(cnn_aA)
    
    #Flatten
    fcn_a1 = Flatten()(cnn_aB)
    fcn_a2 = Dense(64, activation='sigmoid')(fcn_a1)
    
    #Sortie du noeud de vision
    output = Dense(N_TABLEAUX, activation='softmax')(fcn_a2)
    
    model = Model(inputs, output)
    optim = keras.optimizers.Adam(lr=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['acc'])

    return model

def main():
#     #######
#     #Sample
#     plt.ioff()
#     gen = genDatasetTableaux(1)
#     while True:
#         img, tab = next(gen)
#         plt.title('Tableau : {0}'.format(DICT_TABLEAUX[LIST_TABLEAUX[np.argmax(tab[0])]]))
#         plt.imshow(array_to_img(img[0]))
#         plt.show()

    #######
    #Train
    model = createTableauxModel()
    model.summary()
    #     model.load_weights('weights_tableaux.h5')
    model.fit_generator(genDatasetTableaux(4), steps_per_epoch=64, epochs=50)
    model.save_weights('weights_tableaux.h5')
    
#    #######
#    #Test Reel
#    #####
#    #Sample pred
#    model  = createTableauxModel()
#    model.load_weights('weights_tableaux.h5')
#    plt.ioff()
#    for i in range(100, 150):
#        x_test = np.array([img_to_array(Image.open('gta_rsrc/testset/{0}_{1}.png'.format(np.random.choice(['x', 'y']), i)))])
#        y      = model.predict(x_test, verbose=1)
#        plt.title(DICT_TABLEAUX[LIST_TABLEAUX[np.argmax(y[0])]])
#        plt.imshow(array_to_img(x_test[0]))
#        plt.show()

if __name__ == '__main__':
    main()
    input('press enter to quit...')