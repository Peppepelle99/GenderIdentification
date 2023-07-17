import numpy as np
from models.regression_models import linear_log

K = 5
def scoreFusion(llrs, all_labels):
                
    llr_fus = []
    labels_fus = []
    idx_numbers = np.arange(llrs.shape[1])
    idx_partitions = []
    sizePartitions = int(llrs.shape[1]/K)
    for i in range(0, llrs.shape[1], sizePartitions):
        idx_partitions.append(list(idx_numbers[i:i+sizePartitions]))
    for i in range(K):

        idx_test = idx_partitions[i]
        idx_train = idx_partitions[0:i] + idx_partitions[i+1:]

        # from lists of lists collapse the elemnts in a single list
        idx_train = sum(idx_train, [])


        # partition the data and labels using the already partitioned indexes
        STR = llrs[:,idx_train]
        STE = llrs[:,idx_test]
        LTRS = all_labels[idx_train]
        LTES = all_labels[idx_test]
        
        
        fus_llr = linear_log.compute_linearLogReg(STR, LTRS, STE, 1e-3, 0.5)
        
        llr_fus.append(fus_llr)
        labels_fus.append(LTES)

    llr_fus = np.hstack(llr_fus)
    labels_fus = np.hstack(labels_fus)

    return llr_fus
