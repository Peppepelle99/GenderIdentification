from preprocessing import gauss_znorm, PCA

types_d = {'gaussianization': False, 
           'Z-normalization': False,
           'PCA': False}

def preproc(DTR, DTE, types, pca_value = None):

    
    for i, t in enumerate(types_d):
        types_d[t] = types[i]

    if types_d['gaussianization']:
        DTE= gauss_znorm.Gaussianization(DTR,DTE)
        DTR = gauss_znorm.Gaussianization(DTR,DTR)
                

    if types_d['Z-normalization']:
        DTE = gauss_znorm.z_score_normalization(DTR,DTE)
        DTR = gauss_znorm.z_score_normalization(DTR,DTR)
            
        
    if types_d['PCA']:
        DTE=PCA.PCA(DTR, DTE, pca_value)
        DTR=PCA.PCA(DTR, DTR, pca_value)
    
    return DTR, DTE