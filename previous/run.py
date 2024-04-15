from dataloder import acsincom
from fairlearn.metrics import equalized_odds_difference as eod
from fairlearn.metrics import demographic_parity_difference as dpd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

from model import MCLogisticRegression 

income = acsincom()
state_list = income.statelist

def fairness(target_list, pred_list, pred_acc, s_list):
    ppr_list = []
    tnr_list = []
    tpr_list = []
    converted_s = s_list    #s_list[:, 1]  # sex, 1 attribute
    for s_value in np.unique(converted_s):
        if np.mean(converted_s == s_value) > 0.01: #0.01 <- if the race is less than 1%, igore it.
            indexs0 = np.logical_and(target_list == 0, converted_s == s_value)
            indexs1 = np.logical_and(target_list == 1, converted_s == s_value)
            ppr_list.append(np.mean(pred_list[converted_s == s_value]))
            tnr_list.append(np.mean(pred_acc[indexs0]))
            tpr_list.append(np.mean(pred_acc[indexs1]))
        else:
            print("skipped s", s_value)
    eo_gap = max(max(tnr_list) - min(tnr_list), max(tpr_list) - min(tpr_list))
    dp_gap = max(ppr_list) - min(ppr_list)
    
    return eo_gap, dp_gap

eo_gaps = []
dp_gaps = []
acc_list = []

for state in state_list:
    (x_train, x_test, y_train, y_test, 
    race_train, race_test, sex_train, sex_test) = income.get_data_by_client(state)

    model = MCLogisticRegression(x_train, y_train, x_test, y_test, 
                                     0.1, 200, 2)
    model.train()
    
    y_pred = model.predict(x_test)
    pred_acc = (y_pred == y_test)
    acc = accuracy_score(y_test, y_pred)
    eo_gap, dp_gap = fairness(y_test, y_pred, pred_acc, race_test)
    
    eo_gaps.append(eo_gap)
    dp_gaps.append(dp_gap)
    print(acc)
    acc_list.append(acc)

print(np.mean(eo_gaps), np.mean(dp_gaps))
print(np.mean(acc_list))