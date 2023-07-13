from model_evaluation import bayesEval, kFold
from models.GMM import gmm_diag, gmm_full, gmm_tied, gmm_tiedDiag
import numpy as np
from tqdm import tqdm
from plots.plots import plotHist_GMM


type_d = {'full': gmm_full.GMM_Classifier,
          'diag': gmm_diag.GMM_Classifier_Diag,
          'tied': gmm_tied.GMM_Classifier_TiedFull,
          'tiedDiag': gmm_tiedDiag.GMM_Classifier_TiedDiag}




def GMM_analysis(applications,M_list, types, DTR, LTR, title = '', savefig = '', print_values = False):
    listMinDCF = list()
    for t in types:

        llrs_list = list()

        for et in [None, [0,1,0,0]]:
            for M in tqdm(M_list):
                
                all_llrs, all_labels = kFold.k_fold(DTR, LTR, type_d[t], types = et,  params = [M])  
                llrs_list.append(all_llrs)
        
        for app in applications:
            pi1, Cfn, Cfp = app

            for et in range(2):
                for i, m in enumerate(M_list):
                    llr = llrs_list[i + (len(M_list)*et)]
                    DCF_min =  bayesEval.compute_min_DCF(llr, all_labels, pi1, Cfn, Cfp)

                    if pi1 == 0.5:
                        listMinDCF.append(DCF_min)

                    if print_values:
                        print(f'pi = {pi1}, m = {m}, minDCF = {DCF_min}')
        
            
            
        if print_values == False:
            y1 = listMinDCF[:len(M_list)]
            y2 = listMinDCF[len(M_list):]
            plotHist_GMM(np.arange(len(M_list)),M_list, y1, y2, title, savefig=f'plots/figures/GMM/{savefig}.jpg')