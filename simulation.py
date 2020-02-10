# -*- coding: utf-8 -*-
"""
@author: Louis
"""

import numpy as np
import matplotlib.pyplot as plt
from   game  import Recall, GameGenerator, GameSession, getDefaultGameGenerator
from   agent import Agent

def NoterPolitique(politique):
    MISE_DEPART    = None#1500000
    TAILLE_SESSION = 500#Correspond a ~7H
    NOMBRE_ITER    = 1000

    # dataset = Recall(['logs/session_1.log', 'logs/session_2.log', 'logs/session_3.log'], stochastic=True, infinite=True)
    dataset = getDefaultGameGenerator()

    gains    = [None] * NOMBRE_ITER
    score    = [None] * NOMBRE_ITER
    histRisk = [None] * NOMBRE_ITER
    histOdds = [None] * NOMBRE_ITER

    for i in range(NOMBRE_ITER):
        agent   = Agent(politique, MISE_DEPART)
        session = GameSession(agent, dataset)
        session.run(TAILLE_SESSION, False)

        gains[i], score[i], histRisk[i], histOdds[i] = session.getStats()

        print('{0}/{1}'.format(i+1, NOMBRE_ITER))

    print('GAIN  : Moyenne:{0} / Medianne:{1} / Min:{2} / Max:{3} / Std:{4}'.format(int(np.mean(gains)), int(np.median(gains)), int(np.min(gains)), int(np.max(gains)), int(np.std(gains))))
    print('SCORE : Moyenne:{0} / Medianne:{1} / Min:{2} / Max:{3} / Std:{4}'.format(int(np.mean(score)), int(np.median(score)), int(np.min(score)), int(np.max(score)), int(np.std(score))))
    std = np.std(score)
    moy = np.mean(score)
    plt.title('Score de la politique \'{0}\'_Moy:{1}_Std:{2}'.format(politique, int(moy), int(std)))
    count, bins, C = plt.hist(score, bins=100, density=True)
    plt.plot(bins, 1/(std * np.sqrt(2 * np.pi)) * np.exp(-(bins - moy)**2 / (2 * std**2)), linewidth=2, color='r')
    plt.show()
    print('Graphs des profils de jeu de l\'agent:')
    plt.title('Historique des poids : Normalisé')
    plt.bar(list(range(1, 31)), np.sum(histOdds, axis=0) / np.sum(histOdds))
    plt.show()
    plt.title('Historique des risques : Normalisé')
    plt.bar(list(range(1,  7)), np.sum(histRisk, axis=0) / np.sum(histRisk))
    plt.show()

def alnalyseExhaustive():
    recall = Recall([ 'logs/session_1.log',
                      'logs/session_2.log',
                      'logs/session_3.log'], stochastic=False, infinite=False)
    datas   = np.copy(recall.getOddDatas())
    datas   = np.sort(datas)
    datas   = np.transpose(datas)
    rankval = [None] * 6
    rankpdf = [None] * 6
    for i in range(6):
        unique     = np.unique(datas[i])
        rankval[i] = np.arange(unique[0], unique[-1]+1)
        rankval[i] = np.delete(rankval[i], np.where(rankval[i]==11))
        histo      = np.bincount(datas[i])[rankval[i][0]:rankval[i][-1]+1]
        histo      = np.delete(histo, np.where(histo==0))
        rankpdf[i] = histo / np.sum(histo)

    odds = []
    prob = []
    for n0 in range(len(rankval[0])):
        for n1 in range(len(rankval[1])):
            for n2 in range(len(rankval[2])):
                for n3 in range(len(rankval[3])):
                    for n4 in range(len(rankval[4])):
                        for n5 in range(len(rankval[5])):
                            odds.append([rankval[0][n0], rankval[1][n1], 
                                         rankval[2][n2], rankval[3][n3], 
                                         rankval[4][n4], rankval[5][n5]])
                            prob.append(rankpdf[0][n0] * rankpdf[1][n1]*
                                        rankpdf[2][n2] * rankpdf[3][n3]*
                                        rankpdf[4][n4] * rankpdf[5][n5])
    prob = np.array(prob)
    prob = prob / np.sum(prob)
    
    chvmax = []
    espmax = []
    badodd = []
    posesp = []
    betesp = []
    for n in range(len(odds)):
        #Calculs statistiques
        npOdds = np.array(odds[n])
        mise   = 10000
        pdf    = (1.0/npOdds) / np.sum(1.0/npOdds)
        esp    = mise * (pdf * npOdds - (1.0 - pdf))
        chvmax.append(np.argmax(esp)+1)
        espmax.append(np.max(esp))
        if np.max(esp) < 0.0:
            badodd.append(odds[n])
        elif np.max(esp) > 9000:
            betesp.append(odds[n])
        for m in range(6):
            if esp[m] > 0.0:
                posesp.append(m)
    espmax = np.array(espmax)

#    plt.title('Rang des risques les plus inetrressants.')
#    y, x = np.histogram(chvmax, bins=6, range=(1, 6))
#    plt.bar(range(1, 7), y)
#    plt.show()

#    plt.title('Rang des risques avec une esperance positive.')
#    y, x = np.histogram(posesp, bins=6, range=(1, 6))
#    plt.bar(range(1, 7), y)
#    plt.show()

    Fig   = plt.figure()
    Axe   = plt.subplot(1,1,1)
    Axe.set_title('Map of best expectations.')
    Axe.set_xlabel('Expectation')
    Axe.set_ylabel('Occurence')
    Axe.hist(espmax, bins=1000, density=False)
    Fig.show()
    
    Fig   = plt.figure()
    Axe   = plt.subplot(1,1,1)
    Axe.set_title('Map of weighted best expectations.')
    Axe.set_xlabel('Expectation')
    Axe.set_ylabel('Probability')
    Axe.hist(espmax, bins=1000, weights=prob, density=True)
    Fig.show()    
    y, x = np.histogram(espmax, bins=1000, weights=prob, density=True)
    print('Mean of expectation {0:.0f}'.format(np.mean(x)))
    
##    print('il ne faut pas jouer les poids suivants :')
#    f = open("bad.txt","w+")
#    for case in badodd:
#        f.write('{0}\n'.format(case))
#    f.close()
#    
##    print('il faut absolument jouer les poids suivants :')
#    f = open("best.txt","w+")
#    for case in betesp:
#        f.write('{0}\n'.format(case))
#    f.close()

#    ###########################################################################
#    ## MATRIX PLOT
#    lres  =   1080
#    lmin  = -10000
#    lmax  =  10000
#    npts  = abs(lmin) + abs(lmax)
#    x     = np.arange(lmin, lmax, int(npts/lres) + 1)
#    col   = ('r', 'g', 'b', 'y', 'c', 'm')
#    lines = [None] * 6
#    #Graph dynamique
#    plt.ion()
#    plt.axis([lmin, lmax, 0, 1.0/1000.0])
#    for i in range(6):
#        lines[i], = plt.plot(x, [0.0]*len(x), linewidth=2, color=col[i])
#    plt.show()
#    #Dessine chaques loi normal
#    odds = np.array(odds)
#    np.random.shuffle(odds)
#    for n in range(len(odds)):
#        #Calculs statistiques
#        npOdds = np.array(odds[n])
#        ngame  = 500
#        mise   = 10000
#        pdf    = (1.0/npOdds) / np.sum(1.0/npOdds)
#        esp    = mise * (pdf * npOdds - (1.0 - pdf))
#        var    = (1.0/ngame) * \
#            (np.square(npOdds*mise, dtype=np.float64) * pdf + \
#             np.square(mise, dtype=np.float64) * (1.0 - pdf) - \
#                 np.square(esp, dtype=np.float64))
#        std    = np.sqrt(var, dtype=np.float64)
#        for i in range(6):
#            gauss = (1.0 / (std[i] * np.sqrt(2.0 * np.pi))) * np.exp(-0.5*np.square((x - esp[i]) / std[i]))
#            lines[i].set_ydata(gauss)
#        plt.draw()
#        plt.pause(0.2)


def main():
    # POLITIQUE      = 'RANDOM'
    # POLITIQUE      = 'RISQUE_LVL_1'
    # POLITIQUE      = 'RISQUE_LVL_2'
    # POLITIQUE      = 'RISQUE_LVL_3'
    # POLITIQUE      = 'RISQUE_LVL_4'
    # POLITIQUE      = 'RISQUE_LVL_5'
    # POLITIQUE      = 'RISQUE_LVL_6'
    POLITIQUE      = 'GAUSS'
    
#    dataset = Recall(['logs/session_1.log', 
#                      'logs/session_2.log', 
#                      'logs/session_3.log',
#                      'logs/session_4.log'], stochastic=False, infinite=False)
#    # dataset = Recall(['logs/old.log'], stochastic=False, infinite=False)
#    MISE_DEPART    = 1500000
#    TAILLE_SESSION = dataset.getCount()
#    agent   = Agent(POLITIQUE, MISE_DEPART, verbose=1)
#    session = GameSession(agent, dataset)
#    session.run(TAILLE_SESSION, True)

#    gameGen = getDefaultGameGenerator()
#    MISE_DEPART    = 1500000
#    TAILLE_SESSION = 5000
#    agent   = Agent(POLITIQUE, MISE_DEPART)
#    session = GameSession(agent, gameGen)
#    session.run(TAILLE_SESSION, True)
#
#    # %matplotlib qt
#    NoterPolitique(POLITIQUE)

    alnalyseExhaustive()

if __name__ == '__main__':
    main()
    input('press enter to quit...')