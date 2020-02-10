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

from game import LIST_CHEVEAUX
from game import N_CHEVEAUX, TAILLE_PODIUM

IMG_H         = 600
IMG_W         = 800
N_IMG_PODIUM  = 30
POS_MAP       = [(391,251), (109,252), (673,252)]

POS = [(391,251), (109,252), (673,252)]
DIM = [(12,  24), (11,  20), (11,  20)]

def toOneHot(strTab, vocab):
    #Texte ver OneHot
    onehotData = []
    for char in strTab:
        onehotData.append(vocab.index(char))
    
    data    = np.array(onehotData)
    encoded = np_utils.to_categorical(data, num_classes=len(vocab))
    
    return encoded

def genPodium():
    while True:
        ## Genere l'image de fond
        img = Image.open("gta_rsrc/template_y/{0}.png".format(np.random.randint(0, N_IMG_PODIUM)))
        
        ## Shift les couleurs
        npi = img_to_array(img)
        rvb = (np.random.random(size=3) - 0.4) * 255.0
        npi = np.add(npi, rvb)
        npi = np.clip(npi, 0, 255)
        img = array_to_img(npi)
        
        ## Place les positions
        c_chv = np.random.choice(LIST_CHEVEAUX, TAILLE_PODIUM, replace=False)
        for i in range(TAILLE_PODIUM):
            els = Image.open("gta_rsrc/podium/{0}-{1}.png".format(i+1, c_chv[i]))
            img.paste(els, POS_MAP[i], mask=els)
        
        ## Genere un bruit uniform
        alp = np.random.randint(0, 32)
        col = np.random.randint(50, 200)
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
        
        ## Retourne le resultat
        yield img, c_chv

def genDatasetPodium(N=1):
    gen = genPodium()
   
    while True:
        n_train = N
        x_train = []
        y_train = []
        
        for i in range(n_train):
            x, y = next(gen)
            x_train.append(img_to_array(x))
            y_train.append(toOneHot(y, LIST_CHEVEAUX))

        yield np.array(x_train), np.array(y_train)

def createPodiumModel():
    inputs = Input(shape=(IMG_H, IMG_W, 3))

    #Applique l'algo de vision
    vision = [None] * TAILLE_PODIUM
    for i in range(TAILLE_PODIUM):
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
        vision[i] = Dense(len(LIST_CHEVEAUX), activation='softmax')(fcn_a2)
    
    out_a1 = Concatenate()(vision)
    output = Reshape((TAILLE_PODIUM, len(LIST_CHEVEAUX)))(out_a1)
    
    model = Model(inputs, output)
    optim = keras.optimizers.Adam(lr=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['acc'])

    return model 

def main():
    # #######
    # #Sample
    # plt.ioff()
    # gen = genDatasetPodium(1)
    # while True:
    #     img, wht = next(gen)
    #     legend = 'VEC : '
    #     for i in range(TAILLE_PODIUM):
    #         legend += '{0} '.format(LIST_CHEVEAUX[np.argmax(wht[0,i])])
    #     plt.title(legend)
    #     plt.imshow(array_to_img(img[0]))
    #     plt.show()
    
    # #######
    # #Train
    # model = createPodiumModel()
    # model.summary()
    # # model.load_weights('weights_podium.h5')
    # model.fit_generator(genDatasetPodium(4), steps_per_epoch=64, epochs=80)
    # model.save_weights('weights_podium.h5')
    
    ########
    ##Test
    ######
    ##Sample pred
    #model = createPodiumModel()
    #model.load_weights('weights_podium.h5')
    #t_gene = genDatasetPodium(1)
    #x_test, y_test = next(t_gene)
    #y = model.predict(x_test, verbose=1)
    ##%matplotlib qt
    #legend = 'VEC : '
    #for i in range(TAILLE_PODIUM):
    #    legend += '{0} '.format(LIST_CHEVEAUX[np.argmax(y[0,i])])
    #plt.title(legend)
    #plt.imshow(array_to_img(x_test[0]))
    #plt.show()
    
    #######
    #Test Reel
    #####
    #Sample pred
    model  = createPodiumModel()
    model.load_weights('weights_podium.h5')
    plt.ioff()
    for i in range(300, 616):
        x_test = np.array([img_to_array(Image.open('gta_rsrc/testset/y_{0}.png'.format(i)))])
        y      = model.predict(x_test, verbose=1)
        legend = '{0}.png : '.format(i)
        for j in range(TAILLE_PODIUM):
            legend += '{0} '.format(LIST_CHEVEAUX[np.argmax(y[0,j])])
        plt.title(legend)
        plt.imshow(array_to_img(x_test[0]))
        plt.show()

if __name__ == '__main__':
    main()