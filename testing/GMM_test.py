from model_evaluation import bayesEval
from models.GMM import gmm_diag, gmm_full, gmm_tied, gmm_tiedDiag
from tqdm import tqdm
from testing.utility_test import preproc

type_d = {'full': gmm_full.GMM_Classifier,
          'diag': gmm_diag.GMM_Classifier_Diag,
          'tied': gmm_tied.GMM_Classifier_TiedFull,
          'tiedDiag': gmm_tiedDiag.GMM_Classifier_TiedDiag}




def GMM_test(applications, M_list, types, DTR, LTR, DTE, LTE, eval_types = None):

    if eval_types is not None:
        DTR, DTE = preproc(DTR, DTE, eval_types)

    for t in types:

        llrs_list = list()

        for M in tqdm(M_list):
            test_llrs = type_d[t](DTR, LTR, DTE, M)  
            llrs_list.append(test_llrs)
        
        for app in applications:
            pi1, Cfn, Cfp = app

            for i, m in enumerate(M_list):

                DCF_min =  bayesEval.compute_min_DCF(llrs_list[i], LTE, pi1, Cfn, Cfp)
                DCF_act = bayesEval.compute_act_DCF(llrs_list[i], LTE, pi1, Cfn, Cfp)
                print(f'pi = {pi1}, m = {m}, minDCF = {DCF_min}')
        
            