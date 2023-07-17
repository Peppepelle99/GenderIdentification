from model_evaluation import bayesEval
from models.SVMs import linearSVM, kernelSVM
from testing.utility_test import preproc

type_d = {'linear': linearSVM.train_SVM_linear_2,
          'kernel': kernelSVM.kernel_svm}

def SVM_test(applications, DTR, LTR, DTE, LTE, C, type_k, y = None, eval_types = None, pca_value = None):

    if eval_types is not None:
        DTR, DTE = preproc(DTR, DTE, eval_types, pca_value=pca_value)
         
    if type_k == 'linear':
        llrs = linearSVM.train_SVM_linear_2(DTR, LTR,DTE,C,1)
    elif type_k == 'quadratic':
        llrs = kernelSVM.kernel_svm(DTR, LTR, DTE, C, 'poly')
    else:
        llrs = kernelSVM.kernel_svm(DTR, LTR, DTE, C, 'RBF', y)   
    
    

    for app in applications:
        pi1, Cfn, Cfp = app

        DCF_min =  bayesEval.compute_min_DCF(llrs, LTE, pi1, Cfn, Cfp)
        print(f'pi = {pi1}, c = {C}, minDCF = {DCF_min}')
            
            
        
        
        
    