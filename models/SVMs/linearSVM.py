import numpy as np
import scipy.optimize as so
import utility as ut


def train_SVM_linear_2(DTR, LTR, DTE, C, K, balanced = False, pi1=0.5 ):
    """Implementation of the Linear SVM """
    
    DTREXT = np.vstack([DTR, K* np.ones((1,DTR.shape[1] ))])
    
    Z = np.zeros(LTR.shape)
    Z[LTR == 1] = 1
    Z[LTR == 0] = -1
    
    H = np.dot(DTREXT.T, DTREXT)
    H = ut.vcol(Z)* ut.vrow(Z) * H
    
    def JDual(alpha):
        Ha = np.dot(H,ut.vcol(alpha))
        aHa = np.dot(ut.vrow(alpha), Ha)
        a1 = alpha.sum()
        return -0.5* aHa.ravel() + a1, -Ha.ravel() + np.ones(alpha.size) 
                                       
    def LDual(alpha): 
        loss, grad = JDual(alpha)
        return -loss, -grad 
    
    def JPrimal(w): 
        S = np.dot(ut.vrow(w), DTREXT) 
        loss = np.maximum(np.zeros(S.shape), 1-Z*S).sum()
        return 0.5 *np.linalg.norm(w)*2 + C * loss
    
    bounds = []
    if balanced == False:
        for i in range(DTR.shape[1]):
            bounds.append((0, C))
    elif balanced == True:
        N = LTR.size #tot number of samples
        n_T = (1*(LTR==1)).sum() #num of samples belonging to the true class
        n_F = (1*(LTR==0)).sum() #num of samples belonging to the false class
        pi_emp_T = n_T / N
        pi_emp_F = n_F / N
        C_T = C * pi1 / pi_emp_T
        C_F = C * (1-pi1) / pi_emp_F 
        for i in range(DTR.shape[1]):
            if LTR[i] == 1:
                bounds.append((0,C_T))
            else:
                bounds.append((0,C_F))
 
    alphaStar, _x, _y, = so.fmin_l_bfgs_b( 
        LDual, np.zeros(DTR.shape[1]), bounds = bounds, factr = 0.0, maxiter = 100000, maxfun = 100000
    )
    
    wStar = np.dot(DTREXT, ut.vcol(alphaStar)*ut.vcol(Z))
    
    DTEEXT = np.vstack([DTE, K* np.ones((1,DTE.shape[1] ))])    
     
    S = np.dot(wStar.T, DTEEXT)
    S = np.hstack(ut.vrow(S))
    
    return S


def score_SVM(W, DTE, K):
    DTE = np.vstack((DTE,K*np.ones(DTE.shape[1])))
    
    S = np.dot(W.T, DTE)
    return S

def train_SVM(DTR,LTR, K, C):
    DTR = np.vstack((DTR,K*np.ones(DTR.shape[1])))
    
    Z = np.array([LTR * 2.0 - 1.0])
    H = np.dot(DTR.T,DTR) * np.dot(Z.T, Z)
        
    
    def dual_l(alpha):
        aH = np.dot(alpha.T, H)
        aHa = np.dot(aH, alpha)
        a1 = alpha.sum()
        Loss = 0.5 * aHa - a1
        Grad = np.dot(H, alpha) - 1
        
        return Loss, Grad
    
        
        
    
    alpha = np.zeros(DTR.shape[1])
    Bound = [(0,x) for x in C]
    
    a_opt, dual_l_loss, d = so.fmin_l_bfgs_b(dual_l,alpha, bounds=(Bound), factr=1.0, maxiter=100000, maxfun=100000)    
    W_star = np.dot(DTR, np.array(a_opt*Z).reshape(DTR.shape[1]))
    
    return W_star

def compute_linearSVM(DTR,LTR, DTE, C, K = 1, balanced = False, pi = 0.5):
    
    C = C*np.ones(DTR.shape[1])
    
    if balanced:
        nT = LTR[LTR == 1].size/LTR.size
        C[LTR == 0] = C[LTR == 0] * (1-pi)/(1-nT)
        C[LTR == 1] = C[LTR == 1] * pi/nT
    
    W_star = train_SVM(DTR, LTR, K, C)
    score = score_SVM(W_star, DTE, K)
    
    return score
