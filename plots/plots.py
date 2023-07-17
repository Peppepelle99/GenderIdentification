import matplotlib.pyplot as plt
#from Bayes_Decision_Model_Evaluation import BayesDecision
from numpy.lib.function_base import corrcoef
import numpy as np
import utility as uty


def plot_hist(D, L, save_name=""):
    
    """ Plots histograms given D and L which are training/test data and labels and hFea which are the attributes of the dataset,
        store them in the folder called Generated_figures"""
    
    D0 = D[:, L==0] #male samples
    D1 = D[:, L==1] #female samples

    for dIdx in range(12):
        plt.figure()
        plt.xlabel(f'feature {dIdx}') 
        plt.hist(D0[dIdx, :], bins = 30, density = True, alpha = 0.8, label = 'MALE')
        plt.hist(D1[dIdx, :], bins = 30, density = True, alpha = 0.8, label = 'FEMALE')
        
        plt.legend()
        plt.tight_layout()
        plt.savefig('plots/figures/Histograms/hist_%d_%s.jpg' % (dIdx, save_name), format='jpg')
    
    plt.close()



def covariance(D):
    
    mu = uty.vcol(D.mean(1))
    DC = D - mu
    C = np.dot(DC, DC.T)
    C = C/float(D.shape[1])
    
    return mu, C

def variance(D):
    D_var = []
    
    for i in range(D.shape[0]):
        x = D[i,:]
        mu = x.mean()
        sigma2 = (x-mu)**2
        sigma = sigma2.sum()/x.size
        
        D_var.append(sigma)
    
    return D_var

def pearson_corr_coefficient(cov, varX, varY):
    varX = np.sqrt(varX)
    varY = np.sqrt(varY)
    
    res = np.abs(cov/(varX*varY))
    
    return res

def heat_map(D):
    mu, C = covariance(D)
    D_var = variance(D)
    map_pearson = np.zeros((C.shape))
    
    for i in range(D.shape[0]):
        for j in range(D.shape[0]):
            map_pearson[i,j] = pearson_corr_coefficient(C[i,j], D_var[i], D_var[j])
    
    return map_pearson


def plot_heatmap(D, save_name, color, title = ''):
    
    """ Plots correlations given D which are training/test data, store them in the folder called Generated_figures"""
    
    pearson_matrix = corrcoef(D)
    plt.imshow(pearson_matrix, cmap=color)
    plt.title(title)
    plt.savefig('plots/figures/Correlations/%s.jpg' % (save_name))
    return pearson_matrix

def plotDCFprior(x, y,applications,xlabel,title, savefig = '', type = None):
    
    """ Plots the minDCF trend when the different applications change, x is the list of lambda, y is the list of minDCF,
        store them in the folder called Generated_figures"""
    
    
    plt.figure()
    for i in range(3):
        if type is None:
            plt.semilogx(x, y[i,:], label=f'minDCF(pi1 = {applications[i][0]})')
        else:
            plt.semilogx(x, y[i,:], label=f'minDCF(gamma = -{i+1})')
        
    plt.legend()
    plt.xlim([min(x), max(x)])
    plt.xlabel(xlabel)
    plt.ylabel("min DCF")
    plt.title(title)
    plt.savefig(savefig)
    return

def plotHist_GMM(x,xlabel,y1,y2, title, savefig = ''):
    label1 = 'minDCF(pi1 = 0.5) - raw'
    label2 = 'minDCF(pi1 = 0.5) - z norm'

    plt.figure()
    plt.bar(x-0.2,y1, width=0.3, label = label1)
    plt.bar(x+0.2, y2, width=0.3, label = label2)
    plt.xticks(x, xlabel)
    plt.xlabel('GMM components')
    plt.ylim(0,0.6)
    plt.ylabel('DCF')
    plt.legend()
    plt.title(title)
    plt.savefig(savefig)
    


def bayesError_plot(x, min1, min2, act1, act2, names, title):
    

    plt.plot(x, min1, label=f'{names[0]} - min DCF ', linestyle='--' , color='b')
    plt.plot(x, act1, label=f'{names[0]} - act DCF', color='b')

    plt.plot(x, min2, label=f'{names[1]} - min DCF ', linestyle='--', color='r')
    plt.plot(x, act2, label=f'{names[1]} - act DCF ', color='r')

    plt.xlim([-3, 3])
    plt.xlabel("prior log-odds")
    plt.ylabel("DCF value")
    plt.legend()

    plt.savefig(f'plots/figures/BayesError/{title}.jpg')

def bayesError_plot_test(x, min1, min2, act1, act2, names, title):
    

    plt.plot(x, min1, label=f'{names[0]} - min DCF ', linestyle='--' , color='b')
    plt.plot(x, act1, label=f'{names[0]} - act DCF', color='b')

    plt.plot(x, min2, label=f'{names[1]} - min DCF ', linestyle='--', color='r')
    plt.plot(x, act2, label=f'{names[1]} - act DCF ', color='r')

    plt.xlim([-3, 3])
    plt.xlabel("prior log-odds")
    plt.ylabel("DCF value")
    plt.legend()

    plt.savefig(f'plots/figures/BayesErrorTest/{title}.jpg')