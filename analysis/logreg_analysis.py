from model_evaluation import bayesEval, kFold
from models.regression_models import linear_log, quadratic_log
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from plots.plots import plotDCFprior


type_d = {'linear': linear_log.compute_linearLogReg,
          'quadratic': quadratic_log.compute_quadraticLogReg2}




def Regression_analysis(applications, types, DTR, LTR, lambda_list, eval_types = None, pca_value = None, title = '', print_values = False):
    listMinDCF = list()
    for t in types:

        llrs_list = list()

        for l in tqdm(lambda_list):
            all_llrs, all_labels = kFold.k_fold(DTR, LTR, type_d[t], params = [l, 0.5], types = eval_types, pca_value = pca_value)
            llrs_list.append(all_llrs)

        for app in applications:
            pi1, Cfn, Cfp = app

            for i, l in enumerate(lambda_list):
                DCF_min =  bayesEval.compute_min_DCF(llrs_list[i], all_labels, pi1, Cfn, Cfp)
                listMinDCF.append(DCF_min)

                if print_values:
                    print(f'pi = {pi1}, l = {l}, minDCF = {DCF_min}')
            
            
        if print_values == False:
            min_DCF_plot = np.array(listMinDCF).reshape((len(applications),len(lambda_list)))
            plotDCFprior(lambda_list, min_DCF_plot, applications, '', title, savefig=f'plots/figures/Logreg/minDCF_{title}.jpg' )