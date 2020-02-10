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

from game import LIST_POIDS, N_CHEVEAUX

N_IMG_TAB_PARIS = 30

IMG_H = 600
IMG_W = 800

POS = [(98, 195), (98, 262), (98, 330), (98, 397), (98, 465), (98, 532)]
DIM = [(28,  15), (28,  15), (28,  15), (28,  15), (28,  15), (28,  15)]

def toOneHot(strTab, vocab):
    #Texte ver OneHot
    onehotData = []
    for char in strTab:
        onehotData.append(vocab.index(char))
    
    data    = np.array(onehotData)
    encoded = np_utils.to_categorical(data, num_classes=len(vocab))
    
    return encoded

def genTableauParis():
    while True:
        #Genere l'image de fond
        img = Image.open("gta_rsrc/template_x/{0}.png".format(np.random.randint(0, N_IMG_TAB_PARIS)))
        
        #Place les poids
        wht = [0] * N_CHEVEAUX        
        for i in range(N_CHEVEAUX):
            wht[i] = np.random.choice(LIST_POIDS)
            els    = Image.open("gta_rsrc/poids_x/{0}.png".format(wht[i]))
            els.putalpha(255)
            img.paste(els, POS[i], mask=els)

        #Genere un bruit uniform
        alp = np.random.randint(0, 16)
        col = np.random.randint(150, 200)
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
        
        #Décale l'image
        img = ImageChops.offset(img, np.random.randint(-1, 4), np.random.randint(-2, 3))
        
        #Retourn le resultat
        yield img, wht

def genDatasetParis(N=1):
    gen = genTableauParis()
   
    while True:
        n_train = N
        x_train = []
        y_train = []
        
        for i in range(n_train):
            x, y = next(gen)
            x_train.append(img_to_array(x))
            y_train.append(toOneHot(y, LIST_POIDS))

        yield np.array(x_train), np.array(y_train)

def createParisModel():
    inputs = Input(shape=(IMG_H, IMG_W, 3))

    #Applique l'algo de vision
    vision = [None] * N_CHEVEAUX
    for i in range(N_CHEVEAUX):
        #Exctraction du patche
        patche = Cropping2D(cropping=((POS[i][1], IMG_H-(POS[i][1]+DIM[i][1])), (POS[i][0], IMG_W-(POS[i][0]+DIM[i][0]))), data_format='channels_last')(inputs)

        #Réseau CNN
        cnn_a1 = Conv2D(32, kernel_size=3, padding='same', activation=LeakyReLU(), data_format='channels_last')(patche)
        cnn_a2 = Conv2D(32, kernel_size=3, padding='same', activation=LeakyReLU(), data_format='channels_last')(cnn_a1)
        cnn_a3 = BatchNormalization()(cnn_a2)
        cnn_a4 = MaxPooling2D(pool_size=2)(cnn_a3)
        cnn_a5 = Conv2D(16, kernel_size=3, padding='same', activation=LeakyReLU(), data_format='channels_last')(cnn_a4)
        cnn_a6 = Conv2D(16, kernel_size=3, padding='same', activation=LeakyReLU(), data_format='channels_last')(cnn_a5)
        cnn_a7 = BatchNormalization()(cnn_a6)
        cnn_a8 = MaxPooling2D(pool_size=2)(cnn_a7)
        cnn_a9 = Conv2D( 8, kernel_size=3, padding='same', activation=LeakyReLU(), data_format='channels_last')(cnn_a8)
        cnn_aA = Conv2D( 8, kernel_size=3, padding='same', activation=LeakyReLU(), data_format='channels_last')(cnn_a9)
        cnn_aB = BatchNormalization()(cnn_aA)
        
        #Flatten
        fcn_a1 = Flatten()(cnn_aB)
        fcn_a2 = Dense(64, activation='sigmoid')(fcn_a1)
        
        #Sortie du noeud de vision
        vision[i] = Dense(len(LIST_POIDS), activation='softmax')(fcn_a2)
    
    out_a1 = Concatenate()(vision)
    output = Reshape((N_CHEVEAUX, len(LIST_POIDS)))(out_a1)
    
    model = Model(inputs, output)
    optim = keras.optimizers.Adam(lr=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['acc'])

    return model

def main():
    # #######
    # #Sample
    # plt.ioff()
    # gen = genDatasetParis(1)
    # while True:
    #     img, wht = next(gen)
    #     legend = 'VEC : '
    #     for i in range(N_CHEVEAUX):
    #         legend += '{0} '.format(LIST_POIDS[np.argmax(wht[0,i])])
    #     plt.title(legend)
    #     plt.imshow(array_to_img(img[0]))
    #     plt.show()
    
     #######
     #Train
     model = createParisModel()
     model.summary()
#     model.load_weights('weights_paris.h5')
     model.fit_generator(genDatasetParis(4), steps_per_epoch=64, epochs=100)
     model.save_weights('weights_paris.h5')
    
    ########
    ##Test
    ######
    ##Sample pred
    #model = createParisModel()
    #model.load_weights('weights_paris.h5')
    #t_gene = genDatasetParis(1)
    #x_test, y_test = next(t_gene)
    #y = model.predict(x_test, verbose=1)
    ##%matplotlib qt
    #legend = 'VEC : '
    #for i in range(N_CHEVEAUX):
    #    legend += '{0} '.format(LIST_POIDS[np.argmax(y[0,i])])
    #plt.title(legend)
    #plt.imshow(array_to_img(x_test[0]))
    #plt.show()
    
#    #######
#    #Test Reel
#    #####
#    #Sample pred
#    model  = createParisModel()
#    model.load_weights('weights_paris.h5')
#    plt.ioff()
#    for i in range(250, 615):
#        x_test = np.array([img_to_array(Image.open('gta_rsrc/testset/x_{0}.png'.format(i)))])
#        y      = model.predict(x_test, verbose=1)
#        legend = '{0}.png : '.format(i)
#        for j in range(N_CHEVEAUX):
#            legend += '{0} '.format(LIST_POIDS[np.argmax(y[0,j])])
#        plt.title(legend)
#        plt.imshow(array_to_img(x_test[0]))
#        plt.show()
    
#    #######
#    #Stats Reel
#    #####
#    #Sample pred
#    model  = createParisModel()
#    model.load_weights('weights_paris.h5')
#    preds  = []
#    for i in range(1, 615):
#        x_test = np.array([img_to_array(Image.open('gta_rsrc/testset/x_{0}.png'.format(i)))])
#        y      = model.predict(x_test, verbose=1)
#        paris  = []
#        for j in range(N_CHEVEAUX):
#            paris.append(LIST_POIDS[np.argmax(y[0,j])])
#        preds.append(paris)
#    histo = { 1:0,  2:0,  3:0,  4:0,  5:0,  6:0,  7:0,  8:0,  9:0, 10:0,
#                   12:0, 13:0, 14:0, 15:0, 16:0, 17:0, 18:0, 19:0, 20:0, 
#             21:0, 22:0, 23:0, 24:0, 25:0, 26:0, 27:0, 28:0, 29:0, 30:0}
#    histoplace = {1:dict(histo), 2:dict(histo), 3:dict(histo),  
#                  4:dict(histo), 5:dict(histo), 6:dict(histo)}
#    for pred in preds:
#        cheval = 1
#        for poid in pred:
#            histoplace[cheval][poid] += 1
#            cheval += 1
#    print(histoplace)
#    input('press enter to quit...')

if __name__ == '__main__':
    main()