from model_evaluation import bayesEval, kFold
from models.SVMs import linearSVM, kernelSVM
import numpy as np
from tqdm import tqdm
from plots.plots import plotDCFprior


type_d = {'linear': linearSVM.train_SVM_linear_2,
          'kernel': kernelSVM.kernel_svm}




def SVM_analysis(applications, types, DTR, LTR, C_list, type_k = None, eval_types = None, pca_value = None, title = '', print_values = False):
    listMinDCF = list()
    for t in types:

        llrs_list = list()

        for C in tqdm(C_list):
            if type_k is None:
                all_llrs, all_labels = kFold.k_fold(DTR, LTR, type_d[t], params = [C, 1], types = eval_types, pca_value = pca_value)
            else:
                all_llrs, all_labels = kFold.k_fold(DTR, LTR, type_d[t], params = [C, 'poly'], types = eval_types, pca_value = pca_value)

            llrs_list.append(all_llrs)

        for app in applications:
            pi1, Cfn, Cfp = app

            for i, c in enumerate(C_list):
                DCF_min =  bayesEval.compute_min_DCF(llrs_list[i], all_labels, pi1, Cfn, Cfp)
                listMinDCF.append(DCF_min)

                if print_values:
                    print(f'pi = {pi1}, c = {c}, minDCF = {DCF_min}')
            
            
        if print_values == False:
            min_DCF_plot = np.array(listMinDCF).reshape((len(applications),len(C_list)))
            plotDCFprior(C_list, min_DCF_plot, applications, 'C', title, savefig=f'plots/figures/SVM/{title}.jpg' )

def SVM_RBFanalysis(y_list, DTR, LTR, C_list, type_k = 'RBF', eval_types = None, pca_value = None, title = '', print_values = False):
    listMinDCF = list()

    applications = [(0.5, 1, 1), (0.1, 1, 1), (0.9, 1, 1)]
    
    for y in y_list:
        

        for C in tqdm(C_list):
            for app in applications:
                pi1, Cfn, Cfp = app
                all_llrs, all_labels = kFold.k_fold(DTR, LTR, type_d['kernel'], params = [C, type_k, y], types = eval_types, pca_value = pca_value)
                DCF_min =  bayesEval.compute_min_DCF(all_llrs, all_labels, pi1, Cfn, Cfp)
                DCF_act = bayesEval.compute_act_DCF(all_llrs, all_labels, pi1, Cfn, Cfp)
                listMinDCF.append(DCF_min)

                if print_values:
                    print(f'pi = {pi1}, y = {y}, c = {C}, minDCF = {DCF_min} - actDCF = {DCF_act}')
        
        
    if print_values == False:
        min_DCF_plot = np.array(listMinDCF).reshape((len(y_list),len(C_list)))
        plotDCFprior(C_list, min_DCF_plot, y_list, 'C', title, savefig=f'plots/figures/SVM/{title}.jpg', type='RBF')




