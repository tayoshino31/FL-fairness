import numpy as np
from fairlearn.metrics import equal_opportunity_difference as compute_eo
from fairlearn.metrics import demographic_parity_difference as compute_dp

# def fairness(target_list, pred_list, pred_acc, s_list):
#     ppr_list = []
#     tnr_list = []
#     tpr_list = []
#     converted_s = s_list  #s_list[:, 1]  # sex, 1 attribute
#     for s_value in np.unique(converted_s):
#         if np.mean(converted_s == s_value) > 0: #0.01
#             indexs0 = np.logical_and(target_list == 0, converted_s == s_value)
#             indexs1 = np.logical_and(target_list == 1, converted_s == s_value)
#             ppr_list.append(np.mean(pred_list[converted_s == s_value]))
#             tnr_list.append(np.mean(pred_acc[indexs0]))
#             tpr_list.append(np.mean(pred_acc[indexs1]))
#     eo_gap = max(max(tnr_list) - min(tnr_list), max(tpr_list) - min(tpr_list))
#     dp_gap = max(ppr_list) - min(ppr_list)
#     return eo_gap, dp_gap

def fairness(target_list, pred_list, pred_acc, s_list):
    eo = compute_eo(y_true=target_list,y_pred=pred_list, sensitive_features=s_list)
    dp = compute_dp(y_true=target_list, y_pred=pred_list, sensitive_features=s_list)
    return eo, dp