import numpy as np
from models.regression_models import linear_log
import utility as ut

def scoreCalibration(S, L, param_cal):
        sizePartitions = int(len(S)/5)

        llr_cal = []
        labels_cal = []
        idx_numbers = np.arange(S.size)
        idx_partitions = []
        for i in range(0, S.size, sizePartitions):
            idx_partitions.append(list(idx_numbers[i:i+sizePartitions]))
        for i in range(5):

            idx_test = idx_partitions[i]
            idx_train = idx_partitions[0:i] + idx_partitions[i+1:]

            # from lists of lists collapse the elemnts in a single list
            idx_train = sum(idx_train, [])

            # partition the data and labels using the already partitioned indexes
            STR = S[idx_train]
            STE = S[idx_test]
            LTR = L[idx_train]
            LTE = L[idx_test]
            
            cal_llrs=linear_log.compute_linearLogReg_scoreCal(ut.vrow(STR),LTR,STE,param_cal,0.5)
            llr_cal.append(cal_llrs)
            labels_cal.append(LTE)

        llr_cal = np.hstack(llr_cal)
        labels_cal = np.hstack(labels_cal)

        return llr_cal