import numpy as np
import utility as ut
import scipy.special as ss
import math

def covariance(D):
    
    mu = ut.vcol(D.mean(1))
    DC = D - mu
    C = np.dot(DC, DC.T)
    C = C/float(D.shape[1])
    
    return mu, C

def split(GMM, alpha = 0.1):
    
    size = len(GMM)
    splittedGMM = []
    for i in range(size):
        U, s, Vh = np.linalg.svd(GMM[i][2])
        
        # compute displacement vector
        d = U[:, 0:1] * s[0] ** 0.5 * alpha
        splittedGMM.append((GMM[i][0]/2, GMM[i][1]+d, GMM[i][2]))
        splittedGMM.append((GMM[i][0]/2, GMM[i][1]-d, GMM[i][2]))
        
    return splittedGMM

def constraintCov(C, psi = 0.01):
    U, s, Vh = np.linalg.svd(C)
    s[s < psi] = psi
    C = np.dot(U, ut.vcol(s)*U.T)
    return C

def logpdf_GAU_ND(X,mu, C):
    M = X.shape[0]
    sign, log_abs_C = np.linalg.slogdet(C)
    inv_C = np.linalg.inv(C)
   
    const = - M/2 * np.log( 2 * math.pi)
    const = const - 1/2 * log_abs_C
    
    Y = []
    
    for i in range(X.shape[1]):
        x = X[:, i:i+1]
        res = const - 1/2 * np.dot((x-mu).T, np.dot(inv_C, (x-mu))) ; 
        Y.append(res)
    
    return np.array(Y).ravel()

def logpdf_GMM(X, gmm):
    S=np.zeros((len(gmm), X.shape[1]))
    for i in range(len(gmm)):
        S[i,:]=np.log(gmm[i][0]) + logpdf_GAU_ND(X, gmm[i][1], gmm[i][2])
        
    return ss.logsumexp(S, axis=0)

def EStep(X, G, N, gmm):
    
    SJ = np.zeros((G,N))
    for g in range(G):
        SJ[g,:] = logpdf_GAU_ND(X, gmm[g][1],  gmm[g][2]) + np.log(gmm[g][0])
    SM = ss.logsumexp(SJ, axis = 0)
    llNew = SM.sum()/N
    P = np.exp(SJ-SM)
    
    return P, llNew