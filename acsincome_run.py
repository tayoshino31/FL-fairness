from utils.acsincome.dataloder import acsincom
from utils.fairness import fairness
from sklearn.metrics import accuracy_score
import numpy as np
from fairlearn.metrics import equalized_odds_difference
from fairlearn.metrics import demographic_parity_difference
from models.model import LogisticRegression 


def standalone(data, client_list):
    results = {
        "acc": [],"race_eo": [], "race_dp": [],"sex_eo": [], "sex_dp": []
    }
    
    for client in client_list:
        (x_train, x_test, y_train, y_test, 
        _, race_test, _, sex_test) = data.get_data_by_client(client)
        
        n_features = x_train.shape[1]
        model = LogisticRegression(epochs=200, n_features=n_features)
        model.train(x_train, y_train)
        
        y_pred = model.predict(x_test)
        pred_acc = (y_pred == y_test)
        race_eo, race_dp = fairness(y_test, y_pred, pred_acc, race_test)
        sex_eo, sex_dp = fairness(y_test, y_pred, pred_acc, sex_test)
        
        results["race_eo"].append(race_eo)
        results["race_dp"].append(race_dp)
        results["sex_eo"].append(sex_eo)
        results["sex_dp"].append(sex_dp)
        results["acc"].append(np.mean(pred_acc))
        
    return tuple(np.mean(results[key]) for key in ["acc", "race_eo", "race_dp", "sex_eo", "sex_dp"])


def standalone_global(data, client_list):
    results = {
        "acc": [],"race_eo": [], "race_dp": [],"sex_eo": [], "sex_dp": []
    }
    for client in client_list:
        (x_train, x_test, y_train, y_test, 
        _, race_test, _, sex_test) = data.get_data_by_client(client)
        
        n_features = x_train.shape[1]
        model = LogisticRegression(epochs=200, n_features=n_features)
        model.train(x_train, y_train)
        
        (x_train, x_test, y_train, y_test, 
        _, race_test, _, sex_test) = data.get_data_global()
        y_pred = model.predict(x_test)
        pred_acc = (y_pred == y_test)
        race_eo, race_dp = fairness(y_test, y_pred, pred_acc, race_test)
        sex_eo, sex_dp = fairness(y_test, y_pred, pred_acc, sex_test)
                
        results["race_eo"].append(race_eo)
        results["race_dp"].append(race_dp)
        results["sex_eo"].append(sex_eo)
        results["sex_dp"].append(sex_dp)
        results["acc"].append(np.mean(pred_acc))
        
    return tuple(np.mean(results[key]) for key in ["acc", "race_eo", "race_dp", "sex_eo", "sex_dp"])

def centralized_local(data,client_list):
    results = {
        "acc": [],"race_eo": [], "race_dp": [],"sex_eo": [], "sex_dp": []
    }
    
    (x_train, x_test, y_train, y_test, 
    _, race_test, _, sex_test) = data.get_data_global()
    n_features = x_train.shape[1]
    model = LogisticRegression(epochs=200, n_features=n_features)
    model.train(x_train, y_train)
    
    for client in client_list:
        (x_train, x_test, y_train, y_test, 
        _, race_test, _, sex_test) = data.get_data_by_client(client)
        y_pred = model.predict(x_test)
        pred_acc = (y_pred == y_test)
        race_eo, race_dp = fairness(y_test, y_pred, pred_acc, race_test)
        sex_eo, sex_dp = fairness(y_test, y_pred, pred_acc, sex_test)  
        
        results["race_eo"].append(race_eo)
        results["race_dp"].append(race_dp)
        results["sex_eo"].append(sex_eo)
        results["sex_dp"].append(sex_dp)
        results["acc"].append(np.mean(pred_acc))
    
    return tuple(np.mean(results[key]) for key in ["acc", "race_eo", "race_dp", "sex_eo", "sex_dp"])

def centralized_global(data, client_list):
    (x_train, x_test, y_train, y_test, 
    _, race_test, _, sex_test) = data.get_data_global()
    n_features = x_train.shape[1]
    model = LogisticRegression(epochs=200, n_features=n_features)
    model.train(x_train,y_train)
    y_pred = model.predict(x_test)
    pred_acc = (y_pred == y_test)
    acc = accuracy_score(y_test, y_pred)
    race_eo, race_dp = fairness(y_test, y_pred, pred_acc, race_test)
    sex_eo, sex_dp = fairness(y_test, y_pred, pred_acc, sex_test)
        
    return acc, race_eo, race_dp, sex_eo, sex_dp
 
 
def fedavg_local(income,state_list):
    fedavg_weight = None
    fedavg_bias = None
    fedavg_model = None
    epochs = 200
    
    # training
    for i in range(epochs):
        local_weights = []
        local_biases = []
        for state in state_list:
            (x_train, x_test, y_train, y_test, 
            _, race_test, _, sex_test) = income.get_data_by_client(state)
            n_features = x_train.shape[1]
            model = LogisticRegression(epochs=1, n_features=n_features,weight=fedavg_weight, intercept=fedavg_bias)
            model.train(x_train,y_train)
            local_weights.append(model.w.tolist())
            local_biases.append(model.b.tolist())
            fedavg_model = model
        fedavg_weight = np.array(local_weights).mean(axis=0)
        fedavg_bias = np.array(local_biases).mean(axis=0)
        
    #evaludating
    local_acc = [] 
    local_race_eo = []
    local_race_dp = []
    local_sex_eo = []
    local_sex_dp = []
    for state in state_list:
        (x_train, x_test, y_train, y_test, 
            _, race_test, _, sex_test) = income.get_data_by_client(state)
        y_pred = fedavg_model.predict(x_test)
        pred_acc = (y_pred == y_test)
        acc = accuracy_score(y_test, y_pred)
        race_eo_gap, race_dp_gap = fairness(y_test, y_pred, pred_acc, race_test)
        sex_eo_gap, sex_dp_gap = fairness(y_test, y_pred, pred_acc, sex_test)
        local_acc.append(acc)
        local_race_eo.append(race_eo_gap)
        local_race_dp.append(race_dp_gap)
        local_sex_eo.append(sex_eo_gap)
        local_sex_dp.append(sex_dp_gap)
        
    return np.mean(local_acc), np.mean(local_race_eo), np.mean(local_race_dp), np.mean(local_sex_eo), np.mean(local_sex_dp)

def fedavg_global(income,state_list):    
    fedavg_weight = None
    fedavg_bias = None
    fedavg_model = None
    epochs = 200
    
    # training
    for i in range(epochs):
        local_weights = []
        local_biases = []
        for state in state_list:
            (x_train, x_test, y_train, y_test, 
            race_train, race_test, sex_train, sex_test) = income.get_data_by_client(state)
            n_features = x_train.shape[1]
            model = LogisticRegression(epochs=1, n_features=n_features, weight=fedavg_weight, intercept=fedavg_bias)
            model.train(x_train,y_train)
            local_weights.append(model.w.tolist())
            local_biases.append(model.b.tolist())
            fedavg_model = model
        fedavg_weight = np.array(local_weights).mean(axis=0)
        fedavg_bias = np.array(local_biases).mean(axis=0)
        
    #evaludating
    (x_train, x_test, y_train, y_test, 
        _, race_test, _, sex_test) = income.get_data_global()
    y_pred = fedavg_model.predict(x_test)
    pred_acc = (y_pred == y_test)
    acc = accuracy_score(y_test, y_pred)
    race_eo_gap, race_dp_gap = fairness(y_test, y_pred, pred_acc, race_test)
    sex_eo_gap, sex_dp_gap = fairness(y_test, y_pred, pred_acc, sex_test)
        
    return acc, race_eo_gap, race_dp_gap, sex_eo_gap, sex_dp_gap
 

def get_result():
    acc_list = []
    race_eo_list = []
    race_dp_list = []
    sex_eo_list = []
    sex_dp_list = []
    for i in range(5):
        income = acsincom()
        state_list = income.statelist
        acc, race_eo, race_dp, sex_eo, sex_dp = standalone(income, state_list)  
        #acc, race_eo, race_dp, sex_eo, sex_dp = standalone_global(income, state_list) 
        #acc, race_eo, race_dp, sex_eo, sex_dp = centralized_global(income,state_list)  
        #acc, race_eo, race_dp, sex_eo, sex_dp = centralized_local(income,state_list) 
        #acc, race_eo, race_dp, sex_eo, sex_dp = fedavg_global(income,state_list)  
        #acc, race_eo, race_dp, sex_eo, sex_dp = fedavg_local(income,state_list)  
        acc_list.append(acc)
        race_eo_list.append(race_eo)
        race_dp_list.append(race_dp)
        sex_eo_list.append(sex_eo)
        sex_dp_list.append(sex_dp)
        
    print("acc", np.mean(acc_list),     np.std(acc_list))
    print("race_eo", np.mean(race_eo_list), np.std(race_eo_list))
    print("race_dp", np.mean(race_dp_list), np.std(race_dp_list))
    print("sex_eo", np.mean(sex_eo_list),  np.std(sex_eo_list))
    print("sex_do", np.mean(sex_dp_list),  np.std(sex_dp_list))
    
get_result()