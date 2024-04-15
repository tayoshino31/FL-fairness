import numpy as np

def fairness(target_list, pred_list, pred_acc, s_list):
    ppr_list = []
    tnr_list = []
    tpr_list = []
    converted_s = s_list  #s_list[:, 1]  # sex, 1 attribute
    for s_value in np.unique(converted_s):
        if np.mean(converted_s == s_value) > 0.01: #0.01 - if the race is less than 1%, igore it.
            indexs0 = np.logical_and(target_list == 0, converted_s == s_value)
            indexs1 = np.logical_and(target_list == 1, converted_s == s_value)
            ppr_list.append(np.mean(pred_list[converted_s == s_value]))
            tnr_list.append(np.mean(pred_acc[indexs0]))
            tpr_list.append(np.mean(pred_acc[indexs1]))
        else:
            #print("skipped s", s_value)
            pass
    eo_gap = max(max(tnr_list) - min(tnr_list), max(tpr_list) - min(tpr_list))
    dp_gap = max(ppr_list) - min(ppr_list)
    return eo_gap, dp_gap