from model_evaluation import bayesEval, kFold
from models.regression_models import linear_log, quadratic_log
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from plots.plots import plotDCFprior


type_d = {'linear': linear_log.compute_linearLogReg,
          'quadratic': quadratic_log.compute_quadraticLogReg}


def Regression_analysis2(applications, types, DTR, LTR, lambda_list, eval_types = None, pca_value = None, title = '', print_values = False):
    listMinDCF = list()
    for t in types:
        for app in applications:
            pi1, Cfn, Cfp = app

            for l in tqdm(lambda_list):

                all_llrs, all_labels = kFold.k_fold(DTR, LTR, type_d[t], params = [l, 0.5], types = eval_types, pca_value = pca_value)
                DCF_min =  bayesEval.compute_min_DCF(all_llrs, all_labels, pi1, Cfn, Cfp)
                listMinDCF.append(DCF_min)

                if print_values:
                    print(f'pi = {pi1}, l = {l}, minDCF = {DCF_min}')
            
            
        

        min_DCF_plot = np.array(listMinDCF).reshape((len(applications),len(lambda_list)))
        plotDCFprior(lambda_list, min_DCF_plot, applications, '', title)


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
            
            
        

        min_DCF_plot = np.array(listMinDCF).reshape((len(applications),len(lambda_list)))
        plotDCFprior(lambda_list, min_DCF_plot, applications, '', title)


            
            

            










            
# if QUAD_LOGISTIC:
#     for l in lambda_list:

#         print(" quadratic logistic regression with lamb ", l)
#         all_llrs, all_labels = k_fold(DTR, LTR, K, QuadraticLogisticRegression.QuadraticLogisticRegression, [l])
#         DCF_min =  BayesDecision.compute_min_DCF(all_llrs, all_labels, pi1, Cfn, Cfp)
#         DCF_act = BayesDecision.compute_act_DCF(all_llrs, all_labels, pi1, Cfn, Cfp)
#         print("DCF min= ", DCF_min)
#         print("DCF act = ", DCF_act)
#         # listMinDCF.append(DCF_min)
        
#         """
#         #plot bayes error
#         p = numpy.linspace(-3,3,21)
#         plt.plot(p, graphics.bayes_error_plot(p, test_llrs, LTE,minCost=False), color='r')
#         plt.plot(p, graphics.bayes_error_plot(p, test_llrs, LTE,minCost=True), color='b')
#         plt.ylim([0, 1.1])
#         plt.xlim([-3, 3])
#         plt.savefig('Graphics/Generated_figures/DCFPlots/Quadratic LR-minDCF-actDCF.jpg')
#         plt.show()
#         """
        
#         if CALIBRATION:
#             for l2 in lambda_list:
#                 print(" calibration with logistic regression with lamb ", l2)
#                 all_llrs, all_labels = k_fold(DTR, LTR, K, QuadraticLogisticRegression.QuadraticLogisticRegression, [l], l2)
#                 DCF_act = BayesDecision.compute_act_DCF(all_llrs, all_labels, pi1, Cfn, Cfp)
#                 print("DCF calibrated act = ", DCF_act)
                
#         if BALANCING:
#             print(" balancing of the quadratic logistic regression with lamb ", l)
#             all_llrs, all_labels = k_fold(DTR, LTR, K, QuadraticLogisticRegression.QuadraticLogisticRegression, [l, True, 0.5])
#             DCF_min =  BayesDecision.compute_min_DCF(all_llrs, all_labels, pi1, Cfn, Cfp)
#             DCF_act = BayesDecision.compute_act_DCF(all_llrs, all_labels, pi1, Cfn, Cfp)
#             print("DCF min= ", DCF_min)
#             print("DCF act = ", DCF_act)
#             # listMinDCF.append(DCF_min)