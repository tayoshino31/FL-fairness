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
 
 
def fedavg():
    fedavg_acc = None
    fedavg_weight = None
    fedavg_bias = None
    fedavg_race_eo = None
    fedavg_race_dp = None
    fedavg_sex_eo = None
    fedavg_sex_dp = None
    epochs = 400
    
    for i in range(epochs):
        local_acc = []
        local_weights = []
        local_biases = []
        
        local_race_eo = []
        local_race_dp = []
        local_sex_eo = []
        local_sex_dp = []
        for state in state_list:
            (x_train, x_test, y_train, y_test, 
            race_train, race_test, sex_train, sex_test) = income.get_data_by_client(state)

            model = MCLogisticRegression(x_train, y_train, x_test, y_test, 0.1, 1, 2, 
                                         fedavg_weight, fedavg_bias)
            model.train()
            
            y_pred = model.predict(x_test)
            pred_acc = (y_pred == y_test)
            acc = accuracy_score(y_test, y_pred)
            race_eo_gap, race_dp_gap = fairness(y_test, y_pred, pred_acc, race_test)
            sex_eo_gap, sex_dp_gap = fairness(y_test, y_pred, pred_acc, sex_test)
            
            local_acc.append(acc)
            local_weights.append(model.w.tolist())
            local_biases.append(model.b.tolist())
            local_race_eo.append(race_eo_gap)
            local_race_dp.append(race_dp_gap)
            local_sex_eo.append(sex_eo_gap)
            local_sex_dp.append(sex_dp_gap)
        
        fedavg_weight = np.array(local_weights).mean(axis=0)
        fedavg_bias = np.array(local_biases).mean(axis=0)
      
        
        fedavg_acc = np.mean(local_acc)
        fedavg_race_eo = np.mean(local_race_eo)
        fedavg_race_dp = np.mean(local_race_dp)
        fedavg_sex_eo = np.mean(local_sex_eo)
        fedavg_sex_dp = np.mean(local_sex_dp)
        
    print("federated result")
    print("race_eo", fedavg_race_eo, "race_dp",fedavg_race_dp)
    print("sex_eo", fedavg_sex_eo, "sex_dp", fedavg_sex_dp)
    print("acc", fedavg_acc)
    
 
standalone()   
centralized()
fedavg()


