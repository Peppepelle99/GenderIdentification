import numpy as np
import utility as ut
from models.GMM.utility_gmm import covariance, split, constraintCov, EStep, logpdf_GMM

def GMM_EM_Tied(X, gmm, t = 1e-6):
    llNew = None
    llOld = None
    
    G = len(gmm)
    N = X.shape[1]
    

    while llOld is None or llNew - llOld > t:
        llOld = llNew                
        P,llNew = EStep(X, G, N, gmm)
        
        
        gmmNew = []
        
        C_tied = 0 
        for g in range(G):
            gamma = P[g,:]
            Z = gamma.sum()
            F = (ut.vrow(gamma) * X).sum(1)
            S = np.dot(X, (ut.vrow(gamma)*X).T)
            w = Z/N
            mu = ut.vcol(F/Z)
            C = S/Z - np.dot(mu, mu.T)
            C_tied += Z*C
            gmmNew.append((w,mu,C))
            
        C_tied = constraintCov(1/N * C_tied);   
                   
        
        for g in range(G):
            x = gmmNew[g]
            y = list(x)
            y[2] = C_tied
            gmmNew[g]= tuple(y)
            
        gmm = gmmNew
        
    return gmm

def LBG_Tied(GMM, X, components):
    
    iterations = int(np.log2(components))
    
    for i in range(iterations):
        GMM = split(GMM)
        GMM = GMM_EM_Tied(X, GMM)
        
    return GMM

def GMM_Classifier_TiedFull(DTR, LTR, DTE, components, p1 = 0.5):
    SJoint = np.zeros((2,DTE.shape[1]))
    pi = {1: p1, 0: (1-p1)}
    
    C_tied = np.zeros((DTR.shape[0],DTR.shape[0]))     
    for i in [0,1]:
        D = DTR[:, LTR == i]
        mu, C = covariance(D)
        C_tied += D.shape[1]*C
    C_tied = C_tied/DTR.shape[1]
    
    for i in [0,1]:
        mu, C = covariance(DTR[:, LTR == i])
        gmm = LBG_Tied([(1.0,mu,C_tied)], DTR[:,LTR == i], components)
        
        SJoint[i,:] = np.exp(logpdf_GMM(DTE, gmm)) * pi[i]
        
    return SJoint[1,:] - SJoint[0,:]