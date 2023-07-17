from model_evaluation import bayesEval
from models.generative_models import MVG, NaiveBayes, TiedBayes, TiedGaussian
from testing.utility_test import preproc

type_d = {'MVG': MVG.GaussianClassifier,
          'Naive': NaiveBayes.NaiveBayesClassifier,
          'Tied': TiedGaussian.TiedGaussianClassifier,
          'Tied Naive': TiedBayes.TiedNaiveBayes}

def Generative_testing(applications, types, DTR, LTR, DTE,LTE, eval_types = None, pca_value = None):
    for t in types:
        print(f'\n************ {t} ************\n')

        if eval_types is not None:
            DTR, DTE = preproc(DTR, DTE, eval_types, pca_value=pca_value)

        for app in applications:
            pi1, Cfn, Cfp = app

            test_llrs = type_d[t](DTR, LTR, DTE)
            DCF_min =  bayesEval.compute_min_DCF(test_llrs, LTE, pi1, Cfn, Cfp)
            print(f"({pi1}, {Cfn}, {Cfp}) --  DCF_min = {DCF_min}")
        
        