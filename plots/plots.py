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
        plt.xlabel(f'featrues {dIdx}') 
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


def plot_heatmap(D, save_name, color):
    
    """ Plots correlations given D which are training/test data, store them in the folder called Generated_figures"""
    
    pearson_matrix = heat_map(D)
    plt.imshow(pearson_matrix, cmap=color)
    plt.savefig('plots/figures/Correlations/byMe_%s.jpg' % (save_name))
    return pearson_matrix

def plotDCFprior(x, y,xlabel):
    
    """ Plots the minDCF trend when the different applications change, x is the list of lambda, y is the list of minDCF,
        store them in the folder called Generated_figures"""
    
    
    plt.figure()
    plt.plot(x, y[0:len(x)], label='min DCF prior=0.5', color='b')
    plt.plot(x, y[len(x): 2*len(x)], label='min DCF prior=0.1', color='r')
    plt.plot(x, y[2*len(x): 3*len(x)], label='min DCF prior=0.9', color='g')
    plt.xlim([min(x), max(x)])
    plt.xscale("log")
    plt.legend(["min DCF prior=0.5", "min DCF prior=0.1", "min DCF prior=0.9"])
    plt.xlabel(xlabel)
    plt.ylabel("min DCF")
    plt.savefig('Graphics/Generated_figures/DCFPlots/minDCF_%s.jpg' % (xlabel))
    plt.show()
    return

def plotDCFc(x, y,xlabel):
    
    """ Plots the minDCF trend when the different c change, x is the list of C, y is the list of minDCF,
        store them in the folder called Generated_figures"""
    
    
    plt.figure()
    plt.plot(x, y[0:len(x)], label='min DCF prior=0.5 c=0', color='b')
    plt.plot(x, y[len(x): 2*len(x)], label='min DCF prior=0.5 c=1', color='r')
    plt.plot(x, y[2*len(x): 3*len(x)], label='min DCF prior=0.5 c=10', color='g')
    plt.plot(x, y[3*len(x): 4*len(x)], label='min DCF prior=0.5 c=30', color='m')
    plt.xlim([min(x), max(x)])
    plt.xscale("log")
    plt.legend(["min DCF prior=0.5 c=0", "min DCF prior=0.5 c=1", "min DCF prior=0.5 c=10","min DCF prior=0.5 c=30"])
    plt.xlabel(xlabel)
    plt.ylabel("min DCF")
    plt.savefig('Graphics/Generated_figures/DCFPlots/minDCF_%s.jpg' % (xlabel))
    plt.show()
    return

def plotDCFg(x, y,xlabel):
    
    """ Plots the minDCF trend when the different g change, x is the list of C, y is the list of minDCF,
        store them in the folder called Generated_figures"""
    
    
    plt.figure()
    plt.plot(x, y[0:len(x)], label='min DCF prior=0.5 g=1e-5', color='b')
    plt.plot(x, y[len(x): 2*len(x)], label='min DCF prior=0.5 g=1e-4', color='r')
    plt.plot(x, y[2*len(x): 3*len(x)], label='min DCF prior=0.5 g=1e-3', color='g')
    plt.plot(x, y[3*len(x): 4*len(x)], label='min DCF prior=0.5 g=1e-2', color='m')
    plt.xlim([min(x), max(x)])
    plt.xscale("log")
    plt.legend(["min DCF prior=0.5 g=1e-5", "min DCF prior=0.5 g=1e-4", "min DCF prior=0.5 g=1e-3","min DCF prior=0.5 g=1e-2"])
    plt.xlabel(xlabel)
    plt.ylabel("min DCF")
    plt.savefig('Graphics/Generated_figures/DCFPlots/minDCF_%s.jpg' % (xlabel))
    plt.show()
    return

# def bayes_error_plot(pArray,llrs,Labels, minCost=False):
    
#     """ Plots the bayes error, p in the bound, llr and labels are the log likelihood ratio and the class labels respectively,
#         store them in the folder called Generated_figures"""
    
#     y=[]
#     for p in pArray:
#         pi = 1.0/(1.0+numpy.exp(-p))
#         if minCost:
#             y.append(BayesDecision.compute_min_DCF(llrs, Labels, pi, 1, 1))
#         else: 
#             y.append(BayesDecision.compute_act_DCF(llrs, Labels, pi, 1, 1))

#     return numpy.array(y)