import utility as uty
import plots.plots as plots

from analysis.MVG_analysis import Generative_analysis
from analysis.logreg_analysis import Regression_analysis
from analysis.SVM_analysis import SVM_analysis, SVM_RBFanalysis
from analysis.GMM_analysis import GMM_analysis

from preprocessing import  gauss_znorm
import numpy as np

#load data
DTR, LTR, DTE, LTE = uty.load_train_and_test()

#print stats
# print("Total sample of training set: ", DTR.shape[1] )
# print("Total sample of test set: ", DTE.shape[1] )

# print("Total sample of class 0 for training set: ", (LTR==0).sum())
# print("Total sample of class 1 for training set: ", (LTR==1).sum())

# print("Total sample of class 0 for test set: ", (LTE==0).sum())
# print("Total sample of class 1 for test set: ", (LTE==1).sum())


# ******************************** PREPROCESSING ANALISYS ********************************

# """plot histograms"""
# plots.plot_hist(DTR, LTR)

# """Gaussianization of the data"""
# DTR_gauss = gauss_znorm.Gaussianization(DTR, DTR)

# """plot gaussianization histograms"""
# plots.plot_hist(DTR_gauss, LTR, "gauss")

# """Znormalization of the data"""
# DTR_znorm = gauss_znorm.z_score_normalization(DTR)

# """plot znormalization histograms"""
# plots.plot_hist(DTR_znorm, LTR, "znorm")

# DTR0=DTR_gauss[:,LTR==0]
# DTR1=DTR_gauss[:,LTR==1]

# # """plot correlations"""
# plots.plot_heatmap(DTR0, "male_class_correlation_gauss", 'Reds')
# plots.plot_heatmap(DTR1, "female_class_correlation_gauss", 'Blues')
# plots.plot_heatmap(DTR_gauss, "global_correlation_gauss", 'Greys')

applications = [[0.5,1,1] , [0.1,1,1], [0.9,1,1]]


# ******************************** GENERATIVE MODEL ANALYSIS ********************************

# print('------------------------- Raw Data -------------------------')
# Generative_analysis(applications, ['MVG', 'Naive', 'Tied', 'Tied Naive'], DTR, LTR)

# print('------------------------- Raw + PCA 11 -------------------------')
# Generative_analysis(applications, ['MVG', 'Naive', 'Tied', 'Tied Naive'], DTR, LTR, [0,0,1,0], pca_value=11)

# print('------------------------- Raw + PCA 10 -------------------------')
# Generative_analysis(applications, ['MVG', 'Naive', 'Tied', 'Tied Naive'], DTR, LTR, [0,0,1,0], pca_value=10)

# print('------------------------- Gauss Data -------------------------')
# Generative_analysis(applications, ['MVG', 'Naive', 'Tied', 'Tied Naive'], DTR, LTR, [1,0,0,0])

# print('------------------------- Gauss + PCA 11 -------------------------')
# Generative_analysis(applications, ['MVG', 'Naive', 'Tied', 'Tied Naive'], DTR, LTR, [1,0,1,0], pca_value=11)

# print('------------------------- Gauss + PCA 10 -------------------------')
# Generative_analysis(applications, ['MVG', 'Naive', 'Tied', 'Tied Naive'], DTR, LTR, [1,0,1,0], pca_value=10)

# print('------------------------- Z Data -------------------------')
# Generative_analysis(applications, ['MVG', 'Naive', 'Tied', 'Tied Naive'], DTR, LTR, [0,1,0,0])

# print('------------------------- Z + PCA 11 -------------------------')
# Generative_analysis(applications, ['MVG', 'Naive', 'Tied', 'Tied Naive'], DTR, LTR, [0,1,1,0], pca_value=11)

# print('------------------------- Z + PCA 10 -------------------------')
# Generative_analysis(applications, ['MVG', 'Naive', 'Tied', 'Tied Naive'], DTR, LTR, [0,1,1,0], pca_value=10)


# ******************************** REGRESSION ANALYSIS ********************************

l_list = [((1/10**-i) if i<0 else (10**i)) for i in np.linspace(-5, 5,30)]

# print('------------------------- Raw Data - Linear -------------------------')
# Regression_analysis(applications, ['linear'], DTR, LTR, l_list, title='RAW DATA - LINEAR LOGISTIC REGRESSSION')

# print('------------------------- Z Data - Linear -------------------------')
# Regression_analysis(applications, ['linear'], DTR, LTR, l_list, eval_types=[0,1,0,0], title='Z DATA - LINEAR LOGISTIC REGRESSSION')

# print('------------------------- Gauss Data - Linear -------------------------')
# Regression_analysis(applications, ['linear'], DTR, LTR, l_list, eval_types=[1,0,0,0], title='GAUSS DATA - LINEAR LOGISTIC REGRESSSION')

# print('------------------------- Raw + PCA 11 - Linear -------------------------')
# Regression_analysis(applications, ['linear'], DTR, LTR, l_list, eval_types=[0,0,1,0] , title='RAW DATA - LINEAR LOGISTIC REGRESSSION', print_values=True, pca_value = 11)

# print('------------------------- Z + PCA 11 - Linear -------------------------')
# Regression_analysis(applications, ['linear'], DTR, LTR, l_list, eval_types=[0,1,1,0], title='Z DATA - LINEAR LOGISTIC REGRESSSION', print_values=True, pca_value = 11)

# print('------------------------- Gauss + PCA 11 - Linear -------------------------')
# Regression_analysis(applications, ['linear'], DTR, LTR, l_list, eval_types=[1,0,1,0], title='GAUSS DATA - LINEAR LOGISTIC REGRESSSION', print_values=True, pca_value = 11)

# print('------------------------- Raw Data - Quadratic -------------------------')
# Regression_analysis(applications, ['quadratic'], DTR, LTR, l_list, title='RAW DATA - QUADRATIC LOGISTIC REGRESSSION')

# print('------------------------- Z Data - Quadratic -------------------------')
# Regression_analysis(applications, ['quadratic'], DTR, LTR, l_list, eval_types=[0,1,0,0], title='Z DATA - QUADRATIC LOGISTIC REGRESSSION')

# print('------------------------- Gauss Data - Quadratic -------------------------')
# Regression_analysis(applications, ['quadratic'], DTR, LTR, l_list, eval_types=[1,0,0,0], title='GAUSS DATA - QUADRATIC LOGISTIC REGRESSSION')

# print('------------------------- Raw + PCA 11 - Quadratic -------------------------')
# Regression_analysis(applications, ['quadratic'], DTR, LTR, l_list, eval_types=[0,0,1,0] , title='RAW DATA - QUADRATIC LOGISTIC REGRESSSION', print_values=True, pca_value = 11)

# print('------------------------- Z + PCA 11 - Quadratic -------------------------')
# Regression_analysis(applications, ['quadratic'], DTR, LTR, l_list, eval_types=[0,1,1,0], title='Z DATA - QUADRATIC LOGISTIC REGRESSSION', print_values=True, pca_value = 11)

# print('------------------------- Gauss + PCA 11 - Quadratic -------------------------')
# Regression_analysis(applications, ['quadratic'], DTR, LTR, l_list, eval_types=[1,0,1,0], title='GAUSS DATA - QUADRATIC LOGISTIC REGRESSSION', print_values=True, pca_value = 11)


# ******************************** SVM ANALYSIS ********************************

C_list = [((1/10**-i) if i<0 else (10**i)) for i in np.linspace(-3, 2,30)]
gamma_list = [1/np.exp(3) for x in [1, 2, 3]]


# print('------------------------- Raw Data - Linear  -------------------------')
# SVM_analysis(applications, ['linear'], DTR, LTR, C_list, title='RAW DATA - LINEAR SVM', print_values=True)

# print('------------------------- Z Data - Linear   -------------------------')
# SVM_analysis(applications, ['linear'], DTR, LTR, C_list, eval_types=[0,1,0,0], title='Z DATA - LINEAR SVM', print_values='True')

# print('------------------------- Raw Data - Quadratic -------------------------')
# SVM_analysis(applications, ['kernel'], DTR, LTR, C_list, type_k='poly', title='RAW DATA - Quadratic SVM', print_values=True)

# print('------------------------- Z Data - Quadratic -------------------------')
# SVM_analysis(applications, ['kernel'], DTR, LTR, C_list, type_k='poly', eval_types=[0,1,0,0], title='Z DATA - Quadratic SVM', print_values=True)

# print('------------------------- Raw Data - RBF -------------------------')
# SVM_RBFanalysis(gamma_list, DTR, LTR, C_list, type_k='RBF', title='RAW DATA - RBF SVM', print_values=True)

# print('------------------------- Z Data - RBF -------------------------')
# SVM_RBFanalysis(gamma_list, DTR, LTR, C_list, type_k='RBF', eval_types=[0,1,0,0], title='Z DATA - RBF SVM', print_values=True)


# ******************************** GMM ANALYSIS ********************************
M_list = [1,2,4,8,16,32]

#GMM_analysis(applications, M_list, ['full'], DTR, LTR, title='FULL', savefig ='FULL GMM -- RAW-GAUSS')
# GMM_analysis(applications, M_list, ['tied'], DTR, LTR, title='TIED', savefig ='TIED GMM -- RAW-GAUSS')
# GMM_analysis(applications, M_list, ['diag'], DTR, LTR, title='DIAG', savefig ='DIAG GMM -- RAW-GAUSS')
#GMM_analysis(applications, M_list, ['tiedDiag'], DTR, LTR, title='TIED DIAG', savefig ='TIED DIAG GMM -- RAW-GAUSS')

GMM_analysis(applications, M_list, ['full'], DTR, LTR, title='FULL', savefig ='FULL GMM -- RAW-Z')
GMM_analysis(applications, M_list, ['tied'], DTR, LTR, title='TIED', savefig ='TIED GMM -- RAW-Z')
GMM_analysis(applications, M_list, ['diag'], DTR, LTR, title='DIAG', savefig ='DIAG GMM -- RAW-Z')
GMM_analysis(applications, M_list, ['tiedDiag'], DTR, LTR, title='TIED DIAG', savefig ='TIED DIAG GMM -- RAW-Z')
