import numpy as np
from utils.fairness import fairness
from models.model import LogisticRegression 

def get_xys(data):
    x, y, s = [], [], []
    for group in ['A0', 'A1', 'B0', 'B1']:
        x.extend(data[group]['x'])
        y.extend(data[group]['y'])
        s.extend(data[group]['s'])  
    return np.array(x), np.array(y), np.array(s)

def weighted_avg(num_samples, original_weight, original_bias):
    weight = 0.0
    bias = 0.0
    total_samples = np.sum(num_samples)
    for i in range(len(num_samples)):
        weight += (num_samples[i]/total_samples) * original_weight[i]
        bias += (num_samples[i]/total_samples) * original_bias[i]
    return weight, bias

def standalone(x, y , s, lr, epochs): 
    n_features = x.shape[1]
    model = LogisticRegression(epochs=epochs, n_features=n_features, lr=lr)
    model.train(x, y)
    y_pred = model.predict(x)
    pred_acc = (y_pred == y)
    acc = np.mean(pred_acc)
    s_eo, s_dp = fairness(y, y_pred, pred_acc, s)
    decision_boundary = - model.b/ model.w
    return acc, s_eo, s_dp, decision_boundary[0]

def bruteforce(x, y, s, search_range, step, warm_start = None):
    best_decision_boundary = 0
    best_acc = 0
    lower_range = warm_start - search_range
    upper_range = warm_start + search_range
    
    iteration = int((upper_range - lower_range) /step)
    
    for i in range(iteration):
        decision_boundary = lower_range + step * i
        y_pred = [1 if value > decision_boundary else 0 for value in x]
        acc = np.mean((y_pred == y))
        if(best_acc < acc):
            best_decision_boundary = decision_boundary
            best_acc = acc
    #compute fairness 
    y_pred = np.array([1 if value > best_decision_boundary else 0 for value in x])
    pred_acc = (y_pred == y)
    s_eo, s_dp = fairness(y, y_pred, pred_acc, s)
    return best_acc, s_eo, s_dp, best_decision_boundary

def centralized(global_data, lr, epochs):
    x, y, s = get_xys(global_data)
    x = x.reshape(-1,1)
    acc, s_eo, s_dp, decision_boundary = standalone(x, y, s, lr, epochs)
    return acc, s_eo, s_dp, decision_boundary

def fedavg(combined_data, global_data, lr, epochs, local_lrs, local_epochs):
    fedavg_weight = None
    fedavg_bias = None
    #training
    for i in range(epochs):
        local_weights = []
        local_biases = []
        num_samples = [] 
        for idx, client in enumerate(global_data):
            x, y, s = get_xys(client)
            x = x.reshape(-1,1)
            n_features = x.shape[1]    
            client_weight = np.copy(fedavg_weight)
            client_bias = np.copy(fedavg_bias)
            model = LogisticRegression(epochs=int(local_epochs[idx]), 
                                       n_features=n_features, lr=local_lrs[idx],
                                       weight=client_weight, intercept=client_bias, init_params=(i==0))
            model.train(x,y)
            local_weights.append(model.w.tolist())
            local_biases.append(model.b.tolist())
            num_samples.append(len(y))
        fedavg_weight, fedavg_bias = weighted_avg(num_samples, np.array(local_weights), np.array(local_biases))
        #fedavg_weight,fedavg_bias  = np.array(local_weights).mean(axis=0),np.array(local_biases).mean(axis=0)
    
    #eval 
    fedavg_model = LogisticRegression(epochs=int(local_epochs[idx]), n_features=n_features, 
                                      lr=local_lrs[idx], weight=fedavg_weight, intercept=fedavg_bias, init_params=False)
    x, y, s = get_xys(combined_data)
    x = x.reshape(-1,1)
    y_pred = fedavg_model.predict(x)
    pred_acc = (y_pred == y)
    acc = np.mean(pred_acc)
    s_eo, s_dp = fairness(y, y_pred, pred_acc, s)
    decision_boundary = - fedavg_bias/ fedavg_weight
    return acc, s_eo, s_dp, decision_boundary[0]

def eval_local(local_data, decition_boundary):  
    x, y, s = local_data
    y_pred = x > decition_boundary
    pred_acc = (y_pred == y)
    acc = np.sum(pred_acc)/len(x)
    s_eo, s_dp = fairness(y, y_pred, pred_acc, s)
    return acc, s_eo, s_dp, decition_boundary
    