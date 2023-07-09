from model_evaluation import bayesEval, kFold
from models.generative_models import MVG, NaiveBayes, TiedBayes, TiedGaussian

type_d = {'MVG': MVG.GaussianClassifier,
          'Naive': NaiveBayes.NaiveBayesClassifier,
          'Tied': TiedGaussian.TiedGaussianClassifier,
          'Tied Naive': TiedBayes.TiedNaiveBayes}

def Generative_analysis(applications, types, DTR, LTR, eval_types = None, pca_value = None):
    for t in types:
        print(f'\n************ {t} ************\n')
        for app in applications:
            pi1, Cfn, Cfp = app

            all_llrs, all_labels = kFold.k_fold(DTR, LTR, type_d[t], types = eval_types, pca_value = pca_value)
            DCF_min =  bayesEval.compute_min_DCF(all_llrs, all_labels, pi1, Cfn, Cfp)
            print(f"({pi1}, {Cfn}, {Cfp}) --  DCF_min = {DCF_min}")
        
        