# -*- coding: utf-8 -*-
"""
@author: Louis
"""

import numpy as np
import agent as Agent

DICT_TABLEAUX = {0:'TABLEAU_MODE',
                 1:'TABLEAU_PARI',
                 2:'TABLEAU_COURSE',
                 3:'TABLEAU_PODIUM',
                 4:'TABLEAU_INCONNU'}
LIST_CHEVEAUX = [ 1,  2,  3,  4,  5,  6]
LIST_POIDS    = [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10,
                     12, 13, 14, 15, 16, 17, 18, 19, 20,
                 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
LIST_TABLEAUX = list(DICT_TABLEAUX.keys())
N_CHEVEAUX    = len(LIST_CHEVEAUX)
N_POIDS       = len(LIST_POIDS)
TAILLE_PODIUM = 3
N_TABLEAUX    = len(LIST_TABLEAUX)

class Recall:
    def __init__(self, files_path, stochastic=False, infinite=True):
        self._poidsParNum = np.zeros((0,6), dtype=np.uint)
        self._numParPosit = np.zeros((0,3), dtype=np.uint)
        self._partieCount = 0
        self._filesPath   = files_path
        self._stochastic  = stochastic
        self._infinite    = infinite

        if not isinstance(self._filesPath, list):
            raise ValueError('files_path must be an list of str.')

        for file_path in self._filesPath:
            file = open(file_path, 'r')
            out  = ' '
            while out != '':
                self._poidsParNum  = np.append(self._poidsParNum, np.zeros((1,6), dtype=np.uint), axis=0)
                self._numParPosit  = np.append(self._numParPosit, np.zeros((1,3), dtype=np.uint), axis=0)
                self._partieCount += 1

                for line in range(12):
                    out = file.readline()

                    if out == '':
                        self._poidsParNum  = np.delete(self._poidsParNum, -1, axis=0)
                        self._numParPosit  = np.delete(self._numParPosit, -1, axis=0)
                        self._partieCount -= 1
                        break

                    if line == 2:
                        self._poidsParNum[-1, 0] = int(out)
                    elif line == 3:
                        self._poidsParNum[-1, 1] = int(out)
                    elif line == 4:
                        self._poidsParNum[-1, 2] = int(out)
                    elif line == 5:
                        self._poidsParNum[-1, 3] = int(out)
                    elif line == 6:
                        self._poidsParNum[-1, 4] = int(out)
                    elif line == 7:
                        self._poidsParNum[-1, 5] = int(out)

                    if line == 9:
                        self._numParPosit[-1, 0] = int(out)
                    elif line == 10:
                        self._numParPosit[-1, 1] = int(out)
                    elif line == 11:
                        self._numParPosit[-1, 2] = int(out)

            file.close()

    def getCount(self):
        return self._partieCount

    def getOddDatas(self):
        return self._poidsParNum

    def getOddAt(self, index):
        if (index < 1) or (index > self._partieCount):
            raise IndexError('Index {0} out of range. Should be within [1, {1}]'.format(index, self._partieCount));
        
        return self._poidsParNum[index-1]

    def getPodiumAt(self, index):
        if (index < 1) or (index > self._partieCount):
            raise IndexError('Index {0} out of range. Should be within [1, {1}]'.format(index, self._partieCount));
        
        return self._numParPosit[index-1]
    
    def getGenerator(self):
        if (self._stochastic == False) and (self._infinite == False):
            for i in range(1, self.getCount()+1):
                yield self.getOddAt(i), self.getPodiumAt(i)
        
        elif (self._stochastic == False) and (self._infinite == True):
            i = 1
            while True:
                yield self.getOddAt(i), self.getPodiumAt(i)
                i += 1
                if i > self.getCount():
                    i = 1
        
        elif (self._stochastic == True) and (self._infinite == False):
            for i in np.random.permutation(range(1, self.getCount()+1)):
                yield self.getOddAt(i), self.getPodiumAt(i)
        
        elif (self._stochastic == True) and (self._infinite == True):
            while True:
                i = np.random.randint(1, self.getCount()+1)
                yield self.getOddAt(i), self.getPodiumAt(i)

    def __add__(self, other):
        filesPath  = self._filesPath   + other._filesPath
        stochastic = self._stochastic or other._stochastic
        infinite   = self._infinite   or other._infinite
        
        return Recall(files_path=filesPath, stochastic=stochastic, infinite=infinite)

class GameGenerator:
    def __init__(self, recall):
        self._recall = recall
        self._oddVal, self._oddPdf = self._initOddPdf_()
    
    #Genere un profile statistique des données
    def _initOddPdf_(self):
        datas  = np.copy(self._recall.getOddDatas())
        datas  = np.sort(datas)
        datas  = np.transpose(datas)
        
        val = [None] * 6
        pdf = [None] * 6
        for i in range(6):
            unique = np.unique(datas[i])
            val[i] = np.arange(unique[0], unique[-1]+1)
            val[i] = np.delete(val[i], np.where(val[i]==11))
            histo  = np.bincount(datas[i])[val[i][0]:val[i][-1]+1]
            histo  = np.delete(histo, np.where(histo==0))
            pdf[i] = histo / np.sum(histo)

        return val, pdf

    #Genere de nouvelles données a partir des profiles statistiques
    def genarateGame(self, oddsPreset=None):
        #Generer les poids
        if oddsPreset != None:
            odds = np.array(oddsPreset)
        else:
            odds = self.generateOdds()
        
        #Randomise l'ordre des poids
        np.random.shuffle(odds)
        
        podium = self.generatePodium(odds)
        
        return odds, podium

    #Generer un tableau de poids
    def generateOdds(self):
        odds = np.zeros(6)
        for i in range(6):
            odds[i] = np.random.choice(self._oddVal[i], p=self._oddPdf[i])
        return odds

    #Construire le podium a partir des données
    def generatePodium(self, odds):
        return np.random.choice(np.arange(1, 7), p=(1.0/odds)/np.sum(1.0/odds), size=3, replace=False)

    #Recuperer un generateur
    def getGenerator(self, oddsPreset=None):
        while True:
            yield self.genarateGame(oddsPreset)

def getDefaultGameGenerator():
    recall = Recall([ 'logs/session_1.log', 
                      'logs/session_2.log', 
                      'logs/session_3.log',
                      'logs/session_4.log'], stochastic=False, infinite=False)

    return GameGenerator(recall)

class GameSession:
    def __init__(self, agent, generator):
        self._agent     = agent
        self._generator = generator
        self._hist_gainloss = np.array([])
        self._hist_odd      = np.array([])
        self._hist_risk     = np.array([])
    
    def run(self, count, verbose=False):
        self._hist_gainloss = np.zeros(count)
        self._hist_odd      = np.zeros(count)
        self._hist_risk     = np.zeros(count)
        
        genGame = self._generator.getGenerator()
        
        if verbose == True:
            print('Lancement d\'une session de {0} parties.'.format(count))
            print('Politique de l\'agent \'{0}\''.format(self._agent.getPolicy()))
        
        for i in range(count):
            oddtab, podium = next(genGame)
            
            try:
                horse, bet = self._agent.action(oddtab)
            except RuntimeError:
                if verbose == True:
                    print('L\'agent n\'a plus d\'argent. {0} parties jouées.'.format(i))
                break
            
            if self._agent.hasAccount():
                self._agent.addMoney(-bet)
            
            self._hist_gainloss[i] -= bet
            self._hist_odd[i]       = oddtab[horse-1]
            self._hist_risk[i]      = np.sum(np.greater_equal(oddtab[horse-1], oddtab))#1=Peut risqué => 6=Trés risqué
            
            if horse == podium[0]:
                gain = (oddtab[horse-1] + 1) * bet
                if self._agent.hasAccount():
                    self._agent.addMoney(gain)
                self._hist_gainloss[i] += gain
        
        if verbose == True:
            import matplotlib.pyplot as plt
            
            gains, score, histRisk, histOdds = self.getStats()
            print('Gains : {0}'.format(gains))
            print('Score : {0}'.format(score))
            plt.title('Historique des poids')
            plt.bar(list(range(1, 31)), histOdds)
            plt.show()
            plt.title('Historique des risques')
            plt.bar(list(range(1,  7)), histRisk)
            plt.show()
    
    def getStats(self):
        gains         = np.sum(self._hist_gainloss)
        score         = np.mean(self._hist_gainloss)
        histRisk  , P = np.histogram(self._hist_risk, range=(1,  6), bins= 6)
        histOdds  , P = np.histogram(self._hist_odd , range=(1, 30), bins=30)
        
        return gains, score, histRisk, histOdds