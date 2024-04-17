import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split

# https://fairlearn.org/main/user_guide/datasets/acs_income.html#introduction
# https://github.com/socialfoundations/folktables
from folktables import ACSDataSource, ACSIncome, generate_categories
from category_encoding import ACSIncome_categories, STATE_LIST

# class acsincom():
#     def __init__(self):
#         self.data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
#         self.statelist = STATE_LIST
#         self.num_train = 1000
#         self.num_test = 2000
#         self.total = self.num_train + self.num_test
    
#     def get_data_by_client(self, state):   
#         if((state in self.statelist) == False):
#             print("Error: input state is invalid")
#             return 
#         state_data = self.data_source.get_data(states=[state], download=True)
#         x, y, s = ACSIncome.df_to_pandas(state_data, categories=ACSIncome_categories, dummies=True)
#         print(len(x))
#         x, y, s = x[:self.total], y[:self.total].astype(int), s[:self.total]
#         print("nan" , np.sum(x.isna()).to_numpy())
#         x, y, s = x.to_numpy(), y.to_numpy().flatten(), s.to_numpy()
#         race, sex = s[:,0], s[:,1]
#         scaler = StandardScaler()
#         x = scaler.fit_transform(x)  
        
#         (x_train, x_test, y_train, y_test, race_train, race_test, sex_train, sex_test) = train_test_split(x, y, race, sex, train_size= self.num_train,test_size=self.num_test)
        
#         return x_train, x_test, y_train, y_test, race_train, race_test, sex_train, sex_test
    
def get_dummies(df):
    for column, categories in ACSIncome_categories.items():
        if column in df.columns:
            df[column] = pd.Categorical(df[column], categories=categories.keys())
    df_dummies = pd.get_dummies(df, prefix={col: col for col in df.columns if col in ACSIncome_categories})
    return df_dummies

class acsincom():
    def __init__(self):
        self.threshold = 50000
        self.num_train = 1000
        self.num_test = 2000
        data, _ = arff.loadarff('../dataset/ACSIncome/ACSIncome_state_number.arff')
        self.data = pd.DataFrame(data)
        self.statelist = np.unique(data['ST']).tolist()
        self.state_data_dict = dict()
        self.global_data = None
        
        (global_x_train, global_x_test, global_y_train, global_y_test,
         global_race_train, global_race_test, global_sex_train, global_sex_test) = [],[],[],[],[],[],[],[]
        
        for state in self.statelist:
            state_data = self.data[self.data['ST'] == state]
            state_data = state_data[0:(self.num_train + self.num_test)]
            
            #get sensitive varibles
            race = state_data['RAC1P']
            sex = state_data['SEX']

            #process input  #'OCCP','POBP''RELP'
            x = state_data.drop(['PINCP','ST'], axis=1) #should I drop state?
            x = get_dummies(x)
            
            #x = np.array(x.to_numpy() * 1.0)
            #x = x.astype(float)
            
            scaler = StandardScaler()
            x = scaler.fit_transform(x)  
            
            #get target
            y = (state_data['PINCP'] > self.threshold).astype(int)
            
            #split train and test
            x_train, x_test = x[:self.num_train],x[self.num_train:]
            y_train, y_test = y[:self.num_train],y[self.num_train:]
            race_train, race_test = race[:self.num_train],race[self.num_train:]
            sex_train, sex_test = sex[:self.num_train],sex[self.num_train:]
            state_data = [x_train, x_test, y_train, y_test, race_train, race_test, sex_train, sex_test]
            self.state_data_dict[state] = state_data
            
            #global data
            global_x_train.extend(x_train.tolist())
            global_x_test.extend(x_test.tolist())
            global_y_train.extend(y_train)
            global_y_test.extend(y_test)
            global_race_train.extend(race_train)
            global_race_test.extend(race_test)
            global_sex_train.extend(sex_train)
            global_sex_test.extend(sex_test)
            
        self.global_data = (np.array(global_x_train), np.array(global_x_test), np.array(global_y_train), np.array(global_y_test),
                    np.array(global_race_train), np.array(global_race_test), np.array(global_sex_train), np.array(global_sex_test))
    
    def get_data_by_client(self, state):  
        return self.state_data_dict[state]
       
    def get_data_global(self):
        return self.global_data
    
    