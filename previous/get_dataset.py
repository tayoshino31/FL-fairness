from scipy.io import arff
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split


class acsincom():
    # https://fairlearn.org/main/user_guide/datasets/acs_income.html#introduction
    # AGEP: Age as an integer from 0 to 99
    # COW: Class of worker:
    # SCHL: Educational attainment:
    # MAR: Marital status:
    # OCCP: Occupation.
    # POBP: Place of birth.
    # RELP: Relationship to householder:
    # WKHP: Usual hours worked per week in the past 12 months.
    # SEX: Sex code
    # RAC1P: Race code
    # PINCP: Total annual income per person
    def __init__(self, num_data=1664500):
        self.num_data = num_data
    
    def getData(self):      
        num_data = self.num_data
        
        #shuffle rows
        data, _ = arff.loadarff('../dataset/ACSIncome/ACSIncome_state_number.arff')
        data = pd.DataFrame(data)

        #convert y from continuous -> categorical
        y = data[data['PINCP'] > 50000]
        
        y_raw = pd.qcut(data['PINCP'], 2, labels=["Q1", "Q2"])
        data = data.drop(['PINCP'], axis=1)

        #get state
        state = data['ST']

        #one hot encoding
        X_onehot = pd.get_dummies(data, 
                   columns=['COW', 'MAR', 'OCCP','POBP','RELP','SEX','RAC1P']).astype(int)
        X_onehot = X_onehot
        y_onehot = pd.get_dummies(y_raw, columns=['PINCP']).astype(int)
        y_onehot  = np.argmax(y_onehot, axis=1)
        print(X_onehot.shape)

        #Normarization
        scaler = StandardScaler()
        norm_X = scaler.fit_transform(X_onehot)

        #pca
        #pca = PCA()
        #pca.fit(norm_data)
        #pca transform
        #pca = PCA(n_components=10)
        #X_pca = pca.fit_transform(norm_data)
        #print("Explanatory variables:", X_pca.shape)
        
        state_list = state.unique().tolist()
        state_list.sort()
        num_each_st = []
        
        #For Centralized model
        X_all = norm_X
        y_all = y_onehot
        sex_all = np.array(data["SEX"])
        race_all = np.array(data["RAC1P"])
        state_all = []
        
        #For FedAvg model
        X_st = {}
        y_st = {}
        for st in state_list:
            st_indices = state.index[state == st]
            X_st[st] = {"X": norm_X[st_indices],
                        "sex": np.array(data["SEX"][st_indices]),
                        "race": np.array(data["RAC1P"][st_indices])}
            y_st[st] = y_onehot[st_indices]
            state_all.extend([st]*len(y_onehot[st_indices]))
            num_each_st.append(len(st_indices))
            
        return X_st, y_st, X_all, y_all, sex_all, race_all, state_list

# class acsincom():
#     # https://fairlearn.org/main/user_guide/datasets/acs_income.html#introduction
#     # AGEP: Age as an integer from 0 to 99
#     # COW: Class of worker:
#     # SCHL: Educational attainment:
#     # MAR: Marital status:
#     # OCCP: Occupation.
#     # POBP: Place of birth.
#     # RELP: Relationship to householder:
#     # WKHP: Usual hours worked per week in the past 12 months.
#     # SEX: Sex code
#     # RAC1P: Race code
#     # PINCP: Total annual income per person
#     def __init__(self, num_data=1664500):
#         self.num_data = num_data
    
#     def getData(self):      
#         num_data = self.num_data
        
#         #shuffle rows
#         data, _ = arff.loadarff('../dataset/ACSIncome/ACSIncome_state_number.arff')
#         data = pd.DataFrame(data)
#         data = data.sample(frac=1).reset_index(drop=True)
#         data = data[:num_data]

#         #convert y from continuous -> categorical
#         y_raw = pd.qcut(data['PINCP'], 2, labels=["Q1", "Q2"])
#         data = data.drop(['PINCP'], axis=1)
        

#         #get state
#         state = data['ST']

#         #one hot encoding
#         X_onehot = pd.get_dummies(data, 
#                    columns=['COW', 'MAR', 'OCCP','POBP','RELP','SEX','RAC1P']).astype(int)
#         X_onehot = X_onehot
#         y_onehot = pd.get_dummies(y_raw, columns=['PINCP']).astype(int)
#         y_onehot  = np.argmax(y_onehot, axis=1)
#         print(X_onehot.shape)

#         #Normarization
#         scaler = StandardScaler()
#         norm_X = scaler.fit_transform(X_onehot)

#         #pca
#         #pca = PCA()
#         #pca.fit(norm_data)
#         #pca transform
#         #pca = PCA(n_components=10)
#         #X_pca = pca.fit_transform(norm_data)
#         #print("Explanatory variables:", X_pca.shape)
        
#         state_list = state.unique().tolist()
#         state_list.sort()
#         num_each_st = []
        
#         #For Centralized model
#         X_all = norm_X
#         y_all = y_onehot
#         sex_all = np.array(data["SEX"])
#         race_all = np.array(data["RAC1P"])
#         state_all = []
        
#         #For FedAvg model
#         X_st = {}
#         y_st = {}
#         for st in state_list:
#             st_indices = state.index[state == st]
#             X_st[st] = {"X": norm_X[st_indices],
#                         "sex": np.array(data["SEX"][st_indices]),
#                         "race": np.array(data["RAC1P"][st_indices])}
#             y_st[st] = y_onehot[st_indices]
#             state_all.extend([st]*len(y_onehot[st_indices]))
#             num_each_st.append(len(st_indices))
            
#         return X_st, y_st, X_all, y_all, sex_all, race_all, state_list


import subprocess
import json
import os
import numpy as np

class synthetic():
    def __init__(self, sf_rate = 0.5, num_data=100):
        self.num_data = num_data
        self.sf_rate = sf_rate
        
    def getData(self):       
        # Read the data
        new_directory = "leaf-master/data/synthetic"
        try:
            os.chdir(new_directory)
        except:
            pass
        cm1 = "python main.py -num-tasks {} -num-classes 2 -num-dim 5".format(self.num_data)
        cm2 = "./preprocess.sh -s niid --sf 1.0 -k 20 -t sample --tf 0.6"
        try:
            subprocess.run(cm1, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            print(f"Error running command: {e.cmd}")
        try:
            subprocess.run(cm2, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            print(f"Error running command: {e.cmd}")


        data_file = './data/all_data/data.json'
        with open(data_file, 'r') as file:
            data = json.load(file)

        # Access different parts of the data
        user_list = data['users'] 
        num_samples = data['num_samples'] 
        user_data = data['user_data'] 
        
        #for centralized model
        X_user = {}
        y_user = {}
        X_all = []
        y_all = []
        user_id = [] 
        i = 0
        for id in user_list:
            n = num_samples[i]
            X_user[id] = user_data[id]['x']
            y_user[id] = user_data[id]['y']
            X_all.extend(user_data[id]['x'])
            y_all.extend(user_data[id]['y'])
            user_id.extend([id]*n)
            i+=1
        
        
        X_1st = np.array(X_all)[:,0]
        percentile = np.percentile(X_1st, self.sf_rate*100)
        sf_all = np.where(X_1st < percentile, 1, 0)
        
        return X_user, y_user, X_all, y_all, user_list, user_id, sf_all
        #return user_list, num_samples, user_data, X_all, y_all, user_id
        