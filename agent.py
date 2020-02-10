# -*- coding: utf-8 -*-
"""
@author: Louis
"""

import numpy as np
import matplotlib.pyplot as plt
import time

class Agent:    
    POLICY_ENUM = ['RANDOM', 
                   'RISQUE_LVL_1',
                   'RISQUE_LVL_2',
                   'RISQUE_LVL_3',
                   'RISQUE_LVL_4',
                   'RISQUE_LVL_5',
                   'RISQUE_LVL_6',
                   'GAUSS']
    
    def __init__(self, policy, tailleSession=None, money=None, verbose=0):
        self._policy  = policy
        self._money   = money
        self._Ngame   = tailleSession
        self._verbose = verbose
        
        if self._policy not in Agent.POLICY_ENUM:
            raise ValueError('Invalid policy \'{0}\', should be : {1}.'.format(self._policy, Agent.POLICY_ENUM))
        
        #Affiche un graph dynamique
        if self._policy == 'GAUSS' and self._verbose >= 2:
            lres  =   1080
            lmin  = -10000
            lmax  =  10000
            npts  = abs(lmin) + abs(lmax)
            col   = ('r', 'g', 'b', 'y', 'c', 'm') 
            plt.ion()
            self._gaussX     = np.arange(lmin, lmax, int(npts/lres) + 1)
            self._gaussLines = [None] * 6
            self._gaussFig   = plt.figure()
            self._gaussAxe   = plt.subplot(1,1,1)
            self._gaussAxe.set_title('Esperance des poids.')
            self._gaussAxe.set_xlabel('Espérance')
            self._gaussAxe.set_ylabel('Probabilitée normalisée')
            for i in range(6):
                self._gaussLines[i], = self._gaussAxe.plot(self._gaussX, [0.0]*len(self._gaussX), linewidth=2, color=col[i])
            self._gaussFig.show()
    
    def action(self, oddTab):
        mise   = None #Mise entre : [10 .. 10000]
        cheval = None #Choix du cheval : [1 .. 6]
        
        ######################################################################
        ############### RANDOM POLICY
        ##
        if self._policy == 'RANDOM':
            mise   = 10000
            cheval = np.random.randint(1, 6+1)
        
        ######################################################################
        ############### RISQUE_LVL_1 POLICY
        ##
        if self._policy == 'RISQUE_LVL_1':
            cheval  = np.where(oddTab == np.sort(oddTab)[0])
            cheval  = cheval[0].flat[0]
            
            mise    = 10000
            cheval += 1
        
        ######################################################################
        ############### RISQUE_LVL_2 POLICY
        ##
        if self._policy == 'RISQUE_LVL_2':
            cheval  = np.where(oddTab == np.sort(oddTab)[1])
            cheval  = cheval[0].flat[0]
            
            mise    = 10000
            cheval += 1
            
        ######################################################################
        ############### RISQUE_LVL_3 POLICY
        ##
        if self._policy == 'RISQUE_LVL_3':
            cheval  = np.where(oddTab == np.sort(oddTab)[2])
            cheval  = cheval[0].flat[0]
            
            mise    = 10000
            cheval += 1
            
        ######################################################################
        ############### RISQUE_LVL_4 POLICY
        ##
        if self._policy == 'RISQUE_LVL_4':
            cheval  = np.where(oddTab == np.sort(oddTab)[3])
            cheval  = cheval[0].flat[0]
            
            mise    = 10000
            cheval += 1
            
        ######################################################################
        ############### RISQUE_LVL_5 POLICY
        ##
        if self._policy == 'RISQUE_LVL_5':
            cheval  = np.where(oddTab == np.sort(oddTab)[4])
            cheval  = cheval[0].flat[0]
            
            mise    = 10000
            cheval += 1
        
        ######################################################################
        ############### RISQUE_LVL_6 POLICY
        ##
        if self._policy == 'RISQUE_LVL_6':
            cheval  = np.where(oddTab == np.sort(oddTab)[5])
            cheval  = cheval[0].flat[0]
            
            mise    = 10000
            cheval += 1
        
        ######################################################################
        ############### GAUSS POLICY
        ##
        if self._policy == 'GAUSS':      
            mise   = 10000
            
            #Calculs statistiques
            npOdds = np.array(oddTab)
            pdf    = (1.0/npOdds) / np.sum(1.0/npOdds)
            esp    = mise * (pdf * npOdds - (1.0 - pdf))

            #Resume la situation
            if self._verbose >= 1:
                for i in range(6):
                    print('## Stats ##')
                    print('{0}:'.format(i+1))
                    print('  -Esperance : {0:5.0f}$'.format(esp[i]))
                    print('  -Chance    : {0:5.0f}%'.format(pdf[i]*100.0))
            
            #Affichage d'un graphe
            if self._verbose >= 2:
                ngames = self._Ngame if self._Ngame != None else 500
                var    = (1.0/ngames) * \
                    (np.square(npOdds*mise, dtype=np.float64) * pdf + \
                     np.square(mise, dtype=np.float64) * (1.0 - pdf) - \
                         np.square(esp, dtype=np.float64))
                std    = np.sqrt(var, dtype=np.float64)
                
                for i in range(6):
                    gauss = (1.0 / (std[i] * np.sqrt(2.0 * np.pi))) * np.exp(-0.5*np.square((self._gaussX - esp[i]) / std[i]))
                    self._gaussLines[i].set_ydata(gauss)
                    self._gaussLines[i].set_label('Poid {0}'.format(oddTab[i]))
                plt.legend()
                self._gaussAxe.relim()
                self._gaussAxe.autoscale_view()
                self._gaussFig.canvas.flush_events() 
                time.sleep(0.1)
            
            #Selectionne le cheval qui a la meilleur esperance
            cheval = np.argmax(esp).flat[0]+1
        
        if (self._money != None) and (mise > self._money):
            raise RuntimeError('Agent bankruptcy.')
        
        return cheval, mise

    def hasAccount(self):
        return self._money != None

    def addMoney(self, amount):
        if self._money == None:
            raise RuntimeError('Adding money is meaningless when money=None')
        
        self._money += amount

    def getMoney(self):
        return self._money
    
    def getPolicy(self):
        return self._policy
