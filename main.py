import utility as uty
import plots.plots as plots

from analysis.MVG_analysis import Generative_analysis
from analysis.logreg_analysis import Regression_analysis
from analysis.SVM_analysis import SVM_analysis, SVM_RBFanalysis
from analysis.GMM_analysis import GMM_analysis
from analysis.bestModel_analysis import bestModels_bayes, bestModels_fusion

from testing.MVG_test import Generative_testing
from testing.SVM_test import SVM_test
from testing.GMM_test import GMM_test
from testing.best_models_testing import bestModels_bayes_test, bestModels_fusion_test

from preprocessing import  gauss_znorm
import numpy as np

#load data
DTR, LTR, DTE, LTE = uty.load_train_and_test()

# print stats
print("Total sample of training set: ", DTR.shape[1] )
print("Total sample of test set: ", DTE.shape[1] )

print("Total sample of class 0 for training set: ", (LTR==0).sum())
print("Total sample of class 1 for training set: ", (LTR==1).sum())

print("Total sample of class 0 for test set: ", (LTE==0).sum())
print("Total sample of class 1 for test set: ", (LTE==1).sum())


# ******************************** PREPROCESSING ANALISYS ********************************

"""plot histograms"""
plots.plot_hist(DTR, LTR)

"""Gaussianization of the data"""
DTR_gauss = gauss_znorm.Gaussianization(DTR, DTR)

"""plot gaussianization histograms"""
plots.plot_hist(DTR_gauss, LTR, "gauss")

"""Znormalization of the data"""
DTR_znorm = gauss_znorm.z_score_normalization(DTR)

"""plot znormalization histograms"""
plots.plot_hist(DTR_znorm, LTR, "znorm")

DTR0=DTR[:,LTR==0]
DTR1=DTR[:,LTR==1]

# """plot correlations"""
plots.plot_heatmap(DTR0, "male_class_correlation", 'Reds', title='Male')
plots.plot_heatmap(DTR1, "female_class_correlation", 'Blues', title='Female')
plots.plot_heatmap(DTR, "global_correlation", 'Greys', title='All')

applications = [[0.5,1,1] , [0.1,1,1], [0.9,1,1]]


# ******************************** GENERATIVE MODEL ANALYSIS ********************************

print('------------------------- Raw Data -------------------------')
Generative_analysis(applications, ['MVG', 'Naive', 'Tied', 'Tied Naive'], DTR, LTR)

print('------------------------- Raw + PCA 11 -------------------------')
Generative_analysis(applications, ['MVG', 'Naive', 'Tied', 'Tied Naive'], DTR, LTR, [0,0,1], pca_value=11)

print('------------------------- Raw + PCA 10 -------------------------')
Generative_analysis(applications, ['MVG', 'Naive', 'Tied', 'Tied Naive'], DTR, LTR, [0,0,1], pca_value=10)

print('------------------------- Gauss Data -------------------------')
Generative_analysis(applications, ['MVG', 'Naive', 'Tied', 'Tied Naive'], DTR, LTR, [1,0,0])

print('------------------------- Gauss + PCA 11 -------------------------')
Generative_analysis(applications, ['MVG', 'Naive', 'Tied', 'Tied Naive'], DTR, LTR, [1,0,1], pca_value=11)

print('------------------------- Gauss + PCA 10 -------------------------')
Generative_analysis(applications, ['MVG', 'Naive', 'Tied', 'Tied Naive'], DTR, LTR, [1,0,1], pca_value=10)

print('------------------------- Z Data -------------------------')
Generative_analysis(applications, ['MVG', 'Naive', 'Tied', 'Tied Naive'], DTR, LTR, [0,1,0])

print('------------------------- Z + PCA 11 -------------------------')
Generative_analysis(applications, ['MVG', 'Naive', 'Tied', 'Tied Naive'], DTR, LTR, [0,1,1], pca_value=11)

print('------------------------- Z + PCA 10 -------------------------')
Generative_analysis(applications, ['MVG', 'Naive', 'Tied', 'Tied Naive'], DTR, LTR, [0,1,1], pca_value=10)


# ******************************** REGRESSION ANALYSIS ********************************

l_list = [((1/10**-i) if i<0 else (10**i)) for i in np.linspace(-5, 5,30)]

print('------------------------- Raw Data - Linear -------------------------')
Regression_analysis(applications, ['linear'], DTR, LTR, l_list, title='RAW DATA - LINEAR LOGISTIC REGRESSSION', print_values=True)

print('------------------------- Z Data - Linear -------------------------')
Regression_analysis(applications, ['linear'], DTR, LTR, l_list, eval_types=[0,1,0], title='Z DATA - LINEAR LOGISTIC REGRESSSION')

print('------------------------- Gauss Data - Linear -------------------------')
Regression_analysis(applications, ['linear'], DTR, LTR, l_list, eval_types=[1,0,0], title='GAUSS DATA - LINEAR LOGISTIC REGRESSSION')

print('------------------------- Raw + PCA 11 - Linear -------------------------')
Regression_analysis(applications, ['linear'], DTR, LTR, l_list, eval_types=[0,0,1] , title='RAW DATA - LINEAR LOGISTIC REGRESSSION', print_values=True, pca_value = 11)

print('------------------------- Z + PCA 11 - Linear -------------------------')
Regression_analysis(applications, ['linear'], DTR, LTR, l_list, eval_types=[0,1,1], title='Z DATA - LINEAR LOGISTIC REGRESSSION', print_values=True, pca_value = 11)

print('------------------------- Gauss + PCA 11 - Linear -------------------------')
Regression_analysis(applications, ['linear'], DTR, LTR, l_list, eval_types=[1,0,1], title='GAUSS DATA - LINEAR LOGISTIC REGRESSSION', print_values=True, pca_value = 11)

print('------------------------- Raw Data - Quadratic -------------------------')
Regression_analysis(applications, ['quadratic'], DTR, LTR, l_list, title='RAW DATA - QUADRATIC LOGISTIC REGRESSSION')

print('------------------------- Z Data - Quadratic -------------------------')
Regression_analysis(applications, ['quadratic'], DTR, LTR, l_list, eval_types=[0,1,0], title='Z DATA - QUADRATIC LOGISTIC REGRESSSION')

print('------------------------- Gauss Data - Quadratic -------------------------')
Regression_analysis(applications, ['quadratic'], DTR, LTR, l_list, eval_types=[1,0,0], title='GAUSS DATA - QUADRATIC LOGISTIC REGRESSSION')

print('------------------------- Raw + PCA 11 - Quadratic -------------------------')
Regression_analysis(applications, ['quadratic'], DTR, LTR, l_list, eval_types=[0,0,1] , title='RAW DATA - QUADRATIC LOGISTIC REGRESSSION', print_values=True, pca_value = 11)

print('------------------------- Z + PCA 11 - Quadratic -------------------------')
Regression_analysis(applications, ['quadratic'], DTR, LTR, l_list, eval_types=[0,1,1], title='Z DATA - QUADRATIC LOGISTIC REGRESSSION', print_values=True, pca_value = 11)

print('------------------------- Gauss + PCA 11 - Quadratic -------------------------')
Regression_analysis(applications, ['quadratic'], DTR, LTR, l_list, eval_types=[1,0,1], title='GAUSS DATA - QUADRATIC LOGISTIC REGRESSSION', print_values=True, pca_value = 11)


# ******************************** SVM ANALYSIS ********************************

C_list = [((1/10**-i) if i<0 else (10**i)) for i in np.linspace(-3, 2,30)]
gamma_list = [1/np.exp(3)]
C = [10]

print('------------------------- Raw Data - Linear  -------------------------')
SVM_analysis(applications, ['linear'], DTR, LTR, C_list, title='RAW DATA - LINEAR SVM', print_values=True)

print('------------------------- Z Data - Linear   -------------------------')
SVM_analysis(applications, ['linear'], DTR, LTR, C_list, eval_types=[0,1,0,0], title='Z DATA - LINEAR SVM', print_values='True')

print('------------------------- Raw Data - Quadratic -------------------------')
SVM_analysis(applications, ['kernel'], DTR, LTR, C_list, type_k='poly', title='RAW DATA - Quadratic SVM', print_values=True)

print('------------------------- Z Data - Quadratic -------------------------')
SVM_analysis(applications, ['kernel'], DTR, LTR, C_list, type_k='poly', eval_types=[0,1,0], title='Z DATA - Quadratic SVM', print_values=True)

print('------------------------- Raw Data - RBF -------------------------')
SVM_RBFanalysis(gamma_list, DTR, LTR, C_list, type_k='RBF', title='RAW DATA - RBF SVM', print_values=True)

print('------------------------- Z Data - RBF -------------------------')
SVM_RBFanalysis(gamma_list, DTR, LTR, C_list, type_k='RBF', eval_types=[0,1,0,0], title='Z DATA - RBF SVM', print_values=True)


# ******************************** GMM ANALYSIS ********************************
M_list = [1,2,4,8,16,32]

GMM_analysis(applications, M_list, ['full'], DTR, LTR, title='FULL', savefig ='FULL GMM -- RAW-GAUSS')
GMM_analysis(applications, M_list, ['tied'], DTR, LTR, title='TIED', savefig ='TIED GMM -- RAW-GAUSS')
GMM_analysis(applications, M_list, ['diag'], DTR, LTR, title='DIAG', savefig ='DIAG GMM -- RAW-GAUSS')
GMM_analysis(applications, M_list, ['tiedDiag'], DTR, LTR, title='TIED DIAG', savefig ='TIED DIAG GMM -- RAW-GAUSS')

GMM_analysis(applications, M_list, ['full'], DTR, LTR, title='FULL', savefig ='FULL GMM -- RAW-Z')
GMM_analysis(applications, M_list, ['tied'], DTR, LTR, title='TIED', savefig ='TIED GMM -- RAW-Z')
GMM_analysis(applications, M_list, ['diag'], DTR, LTR, title='DIAG', savefig ='DIAG GMM -- RAW-Z')
GMM_analysis(applications, M_list, ['tiedDiag'], DTR, LTR, title='TIED DIAG', savefig ='TIED DIAG GMM -- RAW-Z')


# ******************************** BEST MODELS ANALYSIS ********************************

params = list()
effPriorLogOdds = np.linspace(-3, 3,21)

for p in effPriorLogOdds:
    eff_pi = 1/(1+np.exp(-p))
    params.append((eff_pi, 1, 1))

# GMM - MVG - SVM RBF

print('------------------------- UNCALIBRATED -------------------------')

print('GMM - MVG')
bestModels_bayes(['GMM', 'MVG'], DTR, LTR, 'GMM-MVG', params=params)
print('GMM - SVM')
bestModels_bayes(['GMM', 'SVM'], DTR, LTR, 'GMM-SVM', params=params)

print('------------------------- CALIBRATED -------------------------')

print('GMM - MVG')
bestModels_bayes(['GMM', 'MVG'], DTR, LTR, 'GMM-MVG_calibrated', params=params,  param_cal=0)
print('GMM - SVM')
bestModels_bayes(['GMM', 'SVM'], DTR, LTR, 'GMM-SVM_calibrated', params=params,  param_cal=0)


bestModels_fusion(applications, ['GMM', 'MVG', 'SVM'], DTR, LTR)
bestModels_fusion(applications, ['GMM', 'SVM'], DTR, LTR)


# ******************************** TESTING ********************************

DTR, LTR, DTE, LTE = uty.load_train_and_test()


# ******************************** GENERATIVE MODEL TESTING ********************************

print('------------------------- Raw Data -------------------------')
Generative_testing(applications, ['MVG', 'Naive', 'Tied', 'Tied Naive'], DTR, LTR, DTE, LTE)

print('------------------------- Raw + PCA 11 -------------------------')
Generative_testing(applications, ['MVG', 'Naive', 'Tied', 'Tied Naive'], DTR, LTR, DTE, LTE, [0,0,1], pca_value=11)

print('------------------------- Raw + PCA 10 -------------------------')
Generative_testing(applications, ['MVG', 'Naive', 'Tied', 'Tied Naive'], DTR, LTR, DTE, LTE, [0,0,1], pca_value=10)

print('------------------------- Gauss Data -------------------------')
Generative_testing(applications, ['MVG', 'Naive', 'Tied', 'Tied Naive'], DTR, LTR, DTE, LTE, [1,0,0])

print('------------------------- Gauss + PCA 11 -------------------------')
Generative_testing(applications, ['MVG', 'Naive', 'Tied', 'Tied Naive'], DTR, LTR, DTE, LTE, [1,0,1], pca_value=11)

print('------------------------- Gauss + PCA 10 -------------------------')
Generative_testing(applications, ['MVG', 'Naive', 'Tied', 'Tied Naive'], DTR, LTR, DTE, LTE, [1,0,1], pca_value=10)

print('------------------------- Z Data -------------------------')
Generative_testing(applications, ['MVG', 'Naive', 'Tied', 'Tied Naive'], DTR, LTR, DTE, LTE, [0,1,0])

print('------------------------- Z + PCA 11 -------------------------')
Generative_testing(applications, ['MVG', 'Naive', 'Tied', 'Tied Naive'], DTR, LTR, DTE, LTE, [0,1,1], pca_value=11)

print('------------------------- Z + PCA 10 -------------------------')
Generative_testing(applications, ['MVG', 'Naive', 'Tied', 'Tied Naive'], DTR, LTR, DTE, LTE, [0,1,1], pca_value=10)


# ******************************** SVM TESTING ********************************

C = 1
gamma = 1/np.exp(3)


print('------------------------- Raw Data - Linear  -------------------------')
SVM_test(applications, DTR, LTR, DTE, LTE, C, 'linear')

print('------------------------- Z Data - Linear   -------------------------')
SVM_test(applications, DTR, LTR, DTE, LTE, C, 'linear', eval_types=[0,1,0])

print('------------------------- Raw Data - RBF -------------------------')
SVM_test(applications, DTR, LTR, DTE, LTE, C, 'RBF', y=gamma)

print('------------------------- Z Data - RBF -------------------------')
SVM_test(applications, DTR, LTR, DTE, LTE, C, 'RBF', y=gamma, eval_types=[0,1,0])

print('------------------------- Raw Data - Quadratic -------------------------')
SVM_test(applications, DTR, LTR, DTE, LTE, C, 'quadratic')

print('------------------------- Z Data - Quadratic -------------------------')
SVM_test(applications, DTR, LTR, DTE, LTE, C, 'quadratic', eval_types=[0,1,0])



# ******************************** GMM ANALYSIS ********************************
M_list = [1,2,4,8]

print('------------------------- RAW - FULL -------------------------')
GMM_test(applications, M_list, ['full'], DTR, LTR, DTE, LTE)

print('------------------------- GAUSS - FULL -------------------------')
GMM_test(applications, M_list, ['full'], DTR, LTR, DTE, LTE, eval_types=[1,0,0])

print('------------------------- RAW - TIED -------------------------')
GMM_test(applications, M_list, ['tied'], DTR, LTR, DTE, LTE)

print('------------------------- GAUSS - TIED -------------------------')
GMM_test(applications, M_list, ['tied'], DTR, LTR, DTE, LTE, eval_types=[1,0,0])

print('------------------------- RAW - DIAG -------------------------')
GMM_test(applications, M_list, ['diag'], DTR, LTR, DTE, LTE)

print('------------------------- Gauss - DIAG -------------------------')
GMM_test(applications, M_list, ['diag'], DTR, LTR, DTE, LTE, eval_types=[1,0,0])

print('------------------------- RAW - TIED DIAG -------------------------')
GMM_test(applications, M_list, ['diag'], DTR, LTR, DTE, LTE)

print('------------------------- GAUSS - TIED DIAG -------------------------')
GMM_test(applications, M_list, ['tiedDiag'], DTR, LTR, DTE, LTE, eval_types=[1,0,0])


# ******************************** BEST MODELS ANALYSIS ********************************

params = list()
effPriorLogOdds = np.linspace(-3, 3,21)

for p in effPriorLogOdds:
    eff_pi = 1/(1+np.exp(-p))
    params.append((eff_pi, 1, 1))

print('------------------------- UNCALIBRATED -------------------------')

print('GMM - MVG')
bestModels_bayes_test(['GMM', 'MVG'], DTR, LTR, DTE, LTE, 'GMM-MVG', params=applications, print_values=True)
print('GMM - SVM')
bestModels_bayes_test(['GMM', 'SVM'], DTR, LTR, DTE, LTE, 'GMM-SVM', params=applications, print_values=True)

print('------------------------- CALIBRATED -------------------------')

print('GMM - MVG')
bestModels_bayes_test(['GMM', 'MVG'], DTR, LTR, DTE, LTE, 'GMM-MVG_calibrated', params=applications,  param_cal=0, print_values=True)

print('GMM - SVM')
bestModels_bayes_test(['GMM', 'SVM'], DTR, LTR, DTE, LTE, 'GMM-SVM_calibrated', params=applications,  param_cal=0, print_values=True)


bestModels_fusion_test(applications, ['GMM', 'SVM'], DTR, LTR, DTE, LTE)
bestModels_fusion_test(applications, ['GMM', 'MVG'], DTR, LTR, DTE, LTE)