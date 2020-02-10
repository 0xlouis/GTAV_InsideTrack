# -*- coding: utf-8 -*-
"""
@author: Louis
le jeu doit etre mis en resolution 800x600 en 4:3 et sans bordure
"""

import logging

import pyautogui
import time
import win32gui

import numpy as np

from keras.preprocessing.image import img_to_array

from train_paris import createParisModel
from train_podium import createPodiumModel
from train_tableaux import createTableauxModel

from game import LIST_CHEVEAUX, LIST_POIDS, LIST_TABLEAUX
from game import N_CHEVEAUX, TAILLE_PODIUM, DICT_TABLEAUX

from agent import Agent

COOLDOWN = 0.25

GUI_CHEVAL_TO_XY =  {
                        1:(180,185),
                        2:(180,255),
                        3:(180,320),
                        4:(180,390),
                        5:(180,455),
                        6:(180,525)
                    }

GUI_MISE_TO_CLIC =  {
                        100:0,
                        200:1,
                        300:2,
                        400:3,
                        500:4,
                        600:5,
                        700:6,
                        800:7,
                        900:8,
                        1000:9,
                        1500:10,
                        2000:11,
                        2500:12,
                        3000:13,
                        3500:14,
                        4000:15,
                        4500:16,
                        5000:17,
                        5500:18,
                        6000:19,
                        6500:20,
                        7000:21,
                        7500:22,
                        8000:23,
                        8500:24,
                        9000:25,
                        9500:26,
                        10000:27
                    }

ETAT_ACEUIL         = 0
ETAT_PARI_SIMPLE    = 1
ETAT_FAIRE_JEU      = 2
ETAT_FINIR_JEU      = 3

readyGo = False
winPosX = 0
winPosY = 0
winDimX = 0
winDimY = 0

def decodePoidsFromScreen(screen, ia):
    np_img = np.array([img_to_array(screen)])
    predit = ia.predict(np_img, verbose=0)
    
    poids  = [None] * N_CHEVEAUX
    for i in range(N_CHEVEAUX):
        poids[i] = LIST_POIDS[np.argmax(predit[0,i])]
    
    return poids

def decodePodiumFromScreen(screen, ia):
    np_img = np.array([img_to_array(screen)])
    predit = ia.predict(np_img, verbose=0)
    
    podium = [None] * TAILLE_PODIUM
    for i in range(TAILLE_PODIUM):
        podium[i] = LIST_CHEVEAUX[np.argmax(predit[0,i])]
    
    return podium

def decodeTableauFromScreen(screen, ia):
    np_img = np.array([img_to_array(screen)])
    predit = ia.predict(np_img, verbose=0)
    
    return LIST_TABLEAUX[np.argmax(predit[0])], DICT_TABLEAUX[LIST_TABLEAUX[np.argmax(predit[0])]]

def win32Callback(hwnd, extra):
    global readyGo
    global winPosX
    global winPosY
    global winDimX
    global winDimY
    
    if win32gui.GetWindowText(hwnd) == 'Grand Theft Auto V':
        rect    = win32gui.GetWindowRect(hwnd)
        winPosX = rect[0]
        winPosY = rect[1]
        winDimX = rect[2] - winPosX
        winDimY = rect[3] - winPosY
        readyGo = True

def getWinCibleOffset():
    global readyGo
    readyGo = False
    win32gui.EnumWindows(win32Callback, None)
    while readyGo == False:
        time.sleep(0.05)
    
    if (winDimX != 800) or (winDimY != 600):
        raise ValueError("Le jeu doit etre mis en resolution 800x600 en 4:3 et sans bordure.")

def getScreenShot():
    #Trouve la fenetre
    getWinCibleOffset()
    
    #Prendre la capture
    screen = pyautogui.screenshot(region=(winPosX, winPosY, winDimX, winDimY))
    
    return screen

def waitForTableau(ia, tableau):
    Index_Tableau   = None
    Str_Tableau     = None
    screen          = None
    watch           = None
    
    print('>Attente du tableau {0}'.format(tableau))
    while Str_Tableau != tableau:
        screen = getScreenShot()
        Index_Tableau, Str_Tableau = decodeTableauFromScreen(screen, ia)
        time.sleep(0.5)
        
        if watch != Index_Tableau:
            print(' >Tableau actuel {0}'.format(Str_Tableau))
            watch = Index_Tableau
    
    return screen

#####################################
##############  INIT  ###############
#####################################

## Initialise le loggeur
logFormatter = logging.Formatter("%(message)s")
rootLogger = logging.getLogger()
fileHandler = logging.FileHandler("{0}/{1}.log".format('.', 'latest'))
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)
log = rootLogger
log.setLevel(logging.INFO)

## Initialise les IA
parisModel  = createParisModel()
parisModel.load_weights('weights_paris.h5')
podiumModel = createPodiumModel()
podiumModel.load_weights('weights_podium.h5')
tableauxModel = createTableauxModel()
tableauxModel.load_weights('weights_tableaux.h5')

## Initialise l'agent
agent = Agent(policy='GAUSS', verbose=1)

## Initialise l'actionneur
getWinCibleOffset()

automate        = ETAT_ACEUIL
partie          = 0
chevauxPoids    = None
chevauxPodium   = None
allowControl    = True

#####################################
######    MACHINE A ETAT   ##########
#####################################

while True:
    if automate == ETAT_ACEUIL:
        print('Entrée dans l\'etat : \'{0}\''.format(ETAT_ACEUIL))
        waitForTableau(tableauxModel, 'TABLEAU_MODE')
        
        partie += 1
        log.info('partie {0}'.format(partie))
        
        if allowControl:
            pyautogui.moveTo(winPosX+600, winPosY+485, duration=COOLDOWN)
            pyautogui.mouseDown()
            pyautogui.mouseUp()
        
        waitForTableau(tableauxModel, 'TABLEAU_PARI')
        automate = ETAT_PARI_SIMPLE

    if automate == ETAT_PARI_SIMPLE:
        print('Entrée dans l\'etat : \'{0}\''.format(ETAT_PARI_SIMPLE))
        screen = waitForTableau(tableauxModel, 'TABLEAU_PARI')
        
        #Collecter les datas des six chevaux
        log.info('poids')
        chevauxPoids = decodePoidsFromScreen(screen, parisModel)
        for poid in chevauxPoids:
            log.info(poid)

        #Decide quelle action prendre
        cheval, mise = agent.action(chevauxPoids)

        if allowControl:
            #Selectionner le cheval
            pyautogui.moveTo(winPosX+GUI_CHEVAL_TO_XY[cheval][0], winPosY+GUI_CHEVAL_TO_XY[cheval][1], duration=COOLDOWN)
            pyautogui.mouseDown()
            pyautogui.mouseUp()
    
            #Selectionner la mise
            pyautogui.moveTo(winPosX+600, winPosY+290, duration=COOLDOWN)
            for i in range(GUI_MISE_TO_CLIC[mise]):
                pyautogui.mouseDown()
                pyautogui.mouseUp()
            
            pyautogui.moveTo(winPosX+545, winPosY+440, duration=COOLDOWN)
            pyautogui.mouseDown()
            pyautogui.mouseUp()

        waitForTableau(tableauxModel, 'TABLEAU_COURSE')
        automate = ETAT_FAIRE_JEU

    if automate == ETAT_FAIRE_JEU:
        print('Entrée dans l\'etat : \'{0}\''.format(ETAT_FAIRE_JEU))
        waitForTableau(tableauxModel, 'TABLEAU_PODIUM')
        automate = ETAT_FINIR_JEU

    if automate == ETAT_FINIR_JEU:
        print('Entrée dans l\'etat : \'{0}\''.format(ETAT_FINIR_JEU))
        screen = waitForTableau(tableauxModel, 'TABLEAU_PODIUM')
        
        #Collecter les datas du podium
        log.info('podium')
        chevauxPodium = decodePodiumFromScreen(screen, podiumModel)
        for position in chevauxPodium:
            log.info(position)

        if allowControl:
            pyautogui.moveTo(winPosX+400, winPosY+555, duration=COOLDOWN)
            pyautogui.mouseDown()
            pyautogui.mouseUp()
        
        waitForTableau(tableauxModel, 'TABLEAU_MODE')
        automate = ETAT_ACEUIL
