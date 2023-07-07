import utility as uty
import plots.plots as plots
from MVG_analisys import Generative_analysis
from preprocessing import  gauss_znorm

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

# ******************************** GENERATIVE ANALISYS ********************************

print('------------------------- Raw Data -------------------------')
Generative_analysis(applications, ['MVG', 'Naive', 'Tied', 'Tied Naive'], DTR, LTR)

print('------------------------- Raw + PCA 11 -------------------------')
Generative_analysis(applications, ['MVG', 'Naive', 'Tied', 'Tied Naive'], DTR, LTR, [0,0,1,0], pca_value=11)

print('------------------------- Raw + PCA 10 -------------------------')
Generative_analysis(applications, ['MVG', 'Naive', 'Tied', 'Tied Naive'], DTR, LTR, [0,0,1,0], pca_value=10)

print('------------------------- Gauss -------------------------')
Generative_analysis(applications, ['MVG', 'Naive', 'Tied', 'Tied Naive'], DTR, LTR, [1,0,0,0])

print('------------------------- Gauss + PCA 11 -------------------------')
Generative_analysis(applications, ['MVG', 'Naive', 'Tied', 'Tied Naive'], DTR, LTR, [1,0,1,0], pca_value=11)

print('------------------------- Gauss + PCA 10 -------------------------')
Generative_analysis(applications, ['MVG', 'Naive', 'Tied', 'Tied Naive'], DTR, LTR, [1,0,1,0], pca_value=10)




