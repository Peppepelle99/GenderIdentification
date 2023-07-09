import numpy as np
from preprocessing import gauss_znorm, PCA, LDA

types_d = {'gaussianization': False, 
           'Z-normalization': False,
           'PCA': False,
           'LDA': False}

def k_fold(D, L, algorithm, K=5, params=None, params_cal=None, seed=0, types = None, pca_value = None):
    """ Implementation of the k-fold cross validation approach
        D is the dataset, L the labels, K the number of folds
        algorithm is the algorithm used as classifier
        params are optional additional parameters like hyperparameters
        seed is set to 0 and it's used to randomize partitions
        return: llr and labels
    """
    # each value in "types" vector correspond to value of dict types, in order.
    if types is not None:
        for i, t in enumerate(types_d):
            types_d[t] = types[i]
    
    sizePartitions = int(D.shape[1]/K)
    np.random.seed(seed)

    # permutate the indexes of the samples
    idx_permutation = np.random.permutation(D.shape[1])

    # put the indexes inside different partitions
    idx_partitions = []
    for i in range(0, D.shape[1], sizePartitions):
        idx_partitions.append(list(idx_permutation[i:i+sizePartitions]))

    all_llrs = []
    all_labels = []

    # for each fold, consider the ith partition in the test set
    # the other partitions in the train set
    for i in range(K):
        # keep the i-th partition for test
        # keep the other partitions for train
        idx_test = idx_partitions[i]
        idx_train = idx_partitions[0:i] + idx_partitions[i+1:]

        # from lists of lists collapse the elemnts in a single list
        idx_train = sum(idx_train, [])

        # partition the data and labels using the already partitioned indexes
        DTR = D[:, idx_train]
        DTE = D[:, idx_test]
        LTR = L[idx_train]
        LTE = L[idx_test]

        #apply preprocessing to data
        if types_d['gaussianization']:
            DTE= gauss_znorm.Gaussianization(DTR,DTE)
            DTR = gauss_znorm.Gaussianization(DTR,DTR)
            

        if types_d['Z-normalization']:
                DTE = gauss_znorm.z_score_normalization(DTR,DTE)
                DTR = gauss_znorm.z_score_normalization(DTR,DTR)
                
            
        if types_d['PCA']:
            DTE=PCA.PCA(DTR, DTE, pca_value)
            DTR=PCA.PCA(DTR, DTR, pca_value)
            

        if types_d['LDA']:
            m_lda = 1
            DTE=LDA.LDA(DTR, LTR,DTE, m_lda)
            DTR=LDA.LDA(DTR, LTR,DTR, m_lda)
            
            
        # calculate scores
        if params is not None:
            llr = algorithm(DTR, LTR, DTE, *params)
        else:
            llr = algorithm(DTR, LTR, DTE)
        # add scores and labels for this fold in total
        all_llrs.append(llr)
        all_labels.append(LTE)

    all_llrs = np.hstack(all_llrs)
    all_labels = np.hstack(all_labels)
    
    return all_llrs, all_labels