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





def compute_linearLogReg(DTR, LTR, DTE, l, pi):
    
    v = np.zeros(DTR.shape[0]+1)
    logreg_obj2 = logreg_obj_wrap_balancing(DTR, LTR, l, pi)
    y, _J, _d = so.fmin_l_bfgs_b(logreg_obj2, v, approx_grad=True)
     

    S = np.dot(y[0:-1], DTE) + y[-1]
    
    return S
 

     
     
     