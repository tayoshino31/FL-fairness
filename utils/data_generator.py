import numpy as np 

class Data_Generator():
    def __init__(self, N, n_a, alpha_a, alpha_b, means, stds):
        self.classLabels = ["A1", "A0", "B1", "B0"]
        self.means = means
        self.stds = stds
        self.nums = self.compute_group_nums(N, n_a, alpha_a, alpha_b)
        self.local_data = self.generate_client()
        
    def compute_group_nums(self, N, n_a, alpha_a, alpha_b):
        num_a_pos = N * n_a * alpha_a
        num_a_neg = N * n_a * (1 - alpha_a)
        num_b_pos = N * (1 - n_a) * alpha_b
        num_b_neg = N * (1 - n_a) * (1 - alpha_b)
        return (int(num_a_pos), int(num_a_neg), int(num_b_pos), int(num_b_neg))
        
    def sample_xys(self, n, mu, std, y_val, s_val):
        result = dict()
        result["x"] = np.random.normal(mu, std, n)
        result["y"] = np.ones(n) if y_val == 1 else np.zeros(n)
        result["s"] = np.ones(n) if s_val == 1 else np.zeros(n)
        return result
    
    def generate_client(self):
        local_data = dict(dict())
        for i in range(4):
            label = self.classLabels[i]
            mean = self.means[i]
            std = self.stds[i]
            n = self.nums[i]
            s_val = 1 if label[0] == 'A' else 0
            y_val = 1 if label[1] == '1' else 0
            local_data[label] = self.sample_xys(n, mean, std, y_val, s_val)
        return local_data
    
    def get_client(self):
        return self.local_data
                
    def get_xys(self):
        a1 = self.local_data['A1']
        a0 = self.local_data['A0']
        b1 = self.local_data['B1']
        b0 = self.local_data['B0']
        x = np.concatenate((a1['x'], a0['x'], b1['x'], b0['x'])).reshape(-1,1)
        y = np.concatenate((a1['y'], a0['y'], b1['y'], b0['y']))
        s = np.concatenate((a1['s'], a0['s'], b1['s'], b0['s']))
        return x, y, s