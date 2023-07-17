from model_evaluation import bayesEval, kFold, scoreCalibration, scoreFusion
from models.generative_models import TiedGaussian
from models.SVMs import kernelSVM
from models.GMM import gmm_tied
import numpy as np
from plots.plots import bayesError_plot_test
import utility as ut
from testing.utility_test import preproc


effPriorLogOdds = np.linspace(-3, 3,21)


dcf_dict = {'minDCF0': list(),
            'actDCF0': list(),
            'minDCF1': list(),
            'actDCF1': list()}

def bestModels_bayes_test(types, DTR, LTR,DTE,LTE, title, params, param_cal = None, print_values = False):
    cal_llrs = []

    for idx, t in enumerate(types):

        

        if t == 'MVG':
            all_llrs = TiedGaussian.TiedGaussianClassifier(DTR, LTR, DTE)
        elif t == 'SVM':
            DTR, DTE = preproc(DTR, DTE, [0,1,0])
            all_llrs = kernelSVM.kernel_svm(DTR, LTR, DTE, 10,'RBF', 1/np.exp(3))
        else:
            all_llrs = gmm_tied.GMM_Classifier_TiedFull(DTR, LTR, DTE, 4)

        if param_cal is not None:
            cal_llrs = scoreCalibration.scoreCalibration(all_llrs, LTE, param_cal)

        for p in params:
            pi1, Cfn, Cfp = p


            DCF_min =  bayesEval.compute_min_DCF(all_llrs, LTE, pi1, Cfn, Cfp)

            if param_cal is not None:
                DCF_act = bayesEval.compute_DCF(p, cal_llrs.ravel(), LTE)
            else:
                DCF_act = bayesEval.compute_DCF(p, all_llrs, LTE)
            
            if print_values:
                print(f'pi = {pi1} , model = {t}: minDCF = {DCF_min} - actDCF = {DCF_act}')

            dcf_dict[f'minDCF{idx}'].append(DCF_min)
            dcf_dict[f'actDCF{idx}'].append(DCF_act)
    

    if print_values == False:
        bayesError_plot_test(effPriorLogOdds, dcf_dict['minDCF0'],dcf_dict['minDCF1'], dcf_dict['actDCF0'], dcf_dict['actDCF1'], types, title)

llrs_dict = {
    'MVG': 0,
    'SVM': 0,
    'GMM': 0
}

def bestModels_fusion_test(applications, types, DTR, LTR, DTE, LTE,  param_cal = None):
    

    for t in types:

        if t == 'MVG':
            all_llrs = TiedGaussian.TiedGaussianClassifier(DTR, LTR, DTE)
        elif t == 'SVM':
            DTR, DTE = preproc(DTR, DTE, [0,1,0])
            all_llrs = kernelSVM.kernel_svm(DTR, LTR, DTE, 10,'RBF', 1/np.exp(3))
        else:
            all_llrs = gmm_tied.GMM_Classifier_TiedFull(DTR, LTR, DTE, 4)

        llrs_dict[t] = all_llrs
    
    stacked_llrs = np.vstack((llrs_dict[types[0]],llrs_dict[types[1]]))

    fused_score = scoreFusion.scoreFusion(stacked_llrs, LTE)

    #print(fused_score)
    for app in applications:
        pi1, Cfn, Cfp = app

        DCF_min =  bayesEval.compute_min_DCF(fused_score, LTE, pi1, Cfn, Cfp)
        DCF_act = bayesEval.compute_act_DCF(fused_score, LTE, pi1, Cfn, Cfp)

        print(f"p = {pi1}, fusion : {types[0]} - {types[1]} --  DCF_min = {DCF_min} , DCF_act = {DCF_act}")
        




    


    











