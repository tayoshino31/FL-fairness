# https://fairlearn.org/main/user_guide/datasets/acs_income.html#introduction
# https://github.com/socialfoundations/folktables

import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split
from folktables import ACSDataSource, ACSIncome, generate_categories
from utils.acsincome.category_encoding import ACSIncome_categories, STATE_LIST, get_dummies
    
class acsincom():
    def __init__(self):
        self.threshold = 50000
        self.num_train = 1000
        self.num_test = 2000
        self.load_data()
        self.process_data()
    
    def load_data(self):
        data, _ = arff.loadarff('dataset/ACSIncome/ACSIncome_state_number.arff')
        self.data = pd.DataFrame(data)
        self.statelist = self.data['ST'].unique().tolist()
        
    def process_data(self):
        self.state_data_dict = {}
        self.global_data = {
            'x_train': [],'x_test': [], 'y_train': [],'y_test': [],
            'race_train': [],'race_test': [],'sex_train': [],'sex_test': []
        }
        for state in self.statelist:
            state_data = self.data[self.data['ST'] == state]
            #state_data = state_data.sample(frac=1).reset_index(drop=True)  
            state_data = state_data[0:(self.num_train + self.num_test)]
            state_data = state_data.sample(frac=1).reset_index(drop=True) 
            self.state_data_dict[state] = self.process_state_data(state_data)
        for key in self.global_data:
            self.global_data[key] = np.array(self.global_data[key])
            
    def process_state_data(self, state_data):
        x = state_data.drop(['PINCP', 'ST'], axis=1)
        x = pd.get_dummies(x)
        scaler = StandardScaler()
        x = scaler.fit_transform(x)
        y = (state_data['PINCP'] > self.threshold).astype(int)        
        #split train and test
        x_train, x_test = x[:self.num_train],x[self.num_train:]
        y_train, y_test = y[:self.num_train],y[self.num_train:]
        race_train, race_test = state_data['RAC1P'][:self.num_train],state_data['RAC1P'][self.num_train:]
        sex_train, sex_test = state_data['SEX'][:self.num_train],state_data['SEX'][self.num_train:]
        #store as global data
        for key, value in zip(['x_train', 'x_test', 'y_train', 'y_test', 'race_train', 'race_test', 'sex_train', 'sex_test'],
                            [x_train, x_test, y_train, y_test, race_train, race_test, sex_train, sex_test]):
            self.global_data[key].extend(value if isinstance(value, list) else value.tolist())
        return x_train, x_test, y_train, y_test, race_train, race_test, sex_train, sex_test
        

    def get_data_by_client(self, state):  
        return self.state_data_dict[state]
       
    def get_data_global(self):
        return (
            self.global_data['x_train'], self.global_data['x_test'],
            self.global_data['y_train'], self.global_data['y_test'],
            self.global_data['race_train'], self.global_data['race_test'],
            self.global_data['sex_train'], self.global_data['sex_test']
        )