from dataloder import acsincom
from fairness import fairness
from sklearn.metrics import accuracy_score
import numpy as np

from model import MCLogisticRegression 

income = acsincom()
state_list = income.statelist

def standalone():
    race_eo_gaps = []
    race_dp_gaps = []
    sex_eo_gaps = []
    sex_dp_gaps = []
    acc_list = []
    for state in state_list:
        (x_train, x_test, y_train, y_test, 
        race_train, race_test, sex_train, sex_test) = income.get_data_by_client(state)

        model = MCLogisticRegression(x_train, y_train, x_test, y_test, 0.1, 400, 2)
        model.train()
        
        # model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        pred_acc = (y_pred == y_test)
        acc = accuracy_score(y_test, y_pred)
        race_eo_gap, race_dp_gap = fairness(y_test, y_pred, pred_acc, race_test)
        sex_eo_gap, sex_dp_gap = fairness(y_test, y_pred, pred_acc, sex_test)
        
        race_eo_gaps.append(race_eo_gap)
        race_dp_gaps.append(race_dp_gap)
        sex_eo_gaps.append(sex_eo_gap)
        sex_dp_gaps.append(sex_dp_gap)
        
        acc_list.append(acc)
    print("standalone result")
    print("race_eo", np.mean(race_eo_gaps), "race_dp", np.mean(race_dp_gaps))
    print("sex_eo", np.mean(sex_eo_gaps), "sex_dp", np.mean(sex_dp_gaps))
    print("acc", np.mean(acc_list))
    
standalone()

def centralized():
    (x_train, x_test, y_train, y_test, 
    race_train, race_test, sex_train, sex_test) = income.get_data_global()
    
    model = MCLogisticRegression(x_train, y_train, x_test, y_test, 0.1, 400, 2)
    model.train()
    
    y_pred = model.predict(x_test)
    pred_acc = (y_pred == y_test)
    acc = accuracy_score(y_test, y_pred)
    race_eo_gap, race_dp_gap = fairness(y_test, y_pred, pred_acc, race_test)
    sex_eo_gap, sex_dp_gap = fairness(y_test, y_pred, pred_acc, sex_test)
    print("centralized result")
    print("race_eo", race_eo_gap, "race_dp",race_dp_gap)
    print("sex_eo", sex_eo_gap, "sex_dp", sex_dp_gap)
    print("acc", acc)
    
centralized()


