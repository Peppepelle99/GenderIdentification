import numpy as np;
import scipy.optimize as so;
import utility as ut



def logreg_obj_wrap_balancing(DTR, LTR, l, pi):
    def logreg_obj(v):
        Z = np.zeros(LTR.size)
        Z[LTR == 1] = 1                                                 
        Z[LTR == 0] = -1
        w, b = ut.vcol(v[:-1]), v[-1]  
        s =  np.dot(w.T,DTR) + b         
        J = l/2*np.dot(w.T, w) + (np.logaddexp(0, -s[:,LTR == 1]*Z[LTR == 1])).mean()*(pi) + (np.logaddexp(0, -s[:,LTR == 0]*Z[LTR == 0])).mean()*(1-pi)
        return J.ravel()
    return logreg_obj





def compute_quadraticLogReg(DTR, LTR, DTE, l, pi):

    DTR_exp = [] 
    
    for i in range(DTR.shape[1]):
        x = DTR[:,i:i+1]
        tmp = np.dot(x, x.T)
        tmp2 = np.ravel(tmp, order='F')
        tmp3 = np.concatenate( (ut.vcol(tmp2), ut.vcol(x)), axis=0)
        DTR_exp.append(tmp3)

    DTR_exp=np.hstack(DTR_exp)
    
    DTE_exp = []
    
    for i in range(DTE.shape[1]):
        x = DTE[:,i:i+1]
        tmp = np.dot(x, x.T)
        tmp2 = np.ravel(tmp, order='F')
        tmp3 = np.concatenate( (ut.vcol(tmp2), ut.vcol(x)), axis=0)
        DTE_exp.append(tmp3)

    DTE_exp=np.hstack(DTE_exp)
    
    v = np.zeros(DTR_exp.shape[0]+1)
    logreg_obj2 = logreg_obj_wrap_balancing(DTR_exp, LTR, l, pi)
    y, _J, _d = so.fmin_l_bfgs_b(logreg_obj2, v, approx_grad=True)
     

    S = np.dot(y[0:-1], DTE_exp) + y[-1]
    
    return S


def compute_quadraticLogReg2(DTR, LTR, DTE, l, pi):

    # DTE expanded features
    DTE_ef = np.zeros((DTE.shape[0]**2 + DTE.shape[0], DTE.shape[1]))
    for i in range(DTE.shape[1]):
        x = ut.vcol(DTE[:, i])
        phi = np.dot(x, x.T)
        DTE_ef[:,i] = np.concatenate((phi.ravel(),x.ravel()))
    
    # DTR expanded features
    DTR_ef = np.zeros((DTR.shape[0]**2 + DTR.shape[0], DTR.shape[1]))
    for i in range(DTR.shape[1]):
        x = ut.vcol(DTR[:, i])
        phi = np.dot(x, x.T)
        DTR_ef[:,i] = np.concatenate((phi.ravel(),x.ravel()))
    
    
    v = np.zeros(DTR_ef.shape[0]+1)
    logreg_obj2 = logreg_obj_wrap_balancing(DTR_ef, LTR, l, pi)
    y, _J, _d = so.fmin_l_bfgs_b(logreg_obj2, v, approx_grad=True)
     

    S = np.dot(y[0:-1], DTE_ef) + y[-1]
    
    return S
 