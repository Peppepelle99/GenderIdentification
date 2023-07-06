import numpy as np
import matplotlib.pyplot as plt
import utility as ut
from scipy.stats import norm


def Gaussianization(TD, D):
    """Compute the Gaussianization of the dataset (mapping a set of features to values whose empirical cumulative distribution function
    is well approximated by a Gaussian c.d.f.)"""
    if (TD.shape[0]!=D.shape[0]):
        print("Datasets not aligned in dimensions")
    ranks=[]
    for j in range(D.shape[0]):
        tempSum=0
        for i in range(TD.shape[1]):
            tempSum+=(D[j, :]<TD[j, i]).astype(int)
        tempSum+=1
        ranks.append(tempSum/(TD.shape[1]+2))
    y = norm.ppf(ranks)
    return y



def z_score_normalization(DTR):
    mean = np.mean(DTR, axis=1)  
    std = np.std(DTR, axis=1)    
    normalized_DTR = (DTR - ut.vcol(mean)) / ut.vcol(std)  
    return normalized_DTR