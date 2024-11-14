import numpy as np
class LogisticRegression:
    def __init__(self, epochs, n_features, lr, weight=None, intercept=None, init_params=True):
        self.lr = lr
        self.epochs = epochs        
        if init_params:
            self.w  = np.zeros(n_features)
            self.b = 0
        else:
            self.w = weight
            self.b = intercept
    
    def predict(self, x): 
        z = np.dot(x, self.w) + self.b
        y_prob = 1 /(1 + np.exp(-z))
        y_pred = np.where(y_prob > 0.5, 1, 0 )    
        return y_pred
    
    def accuracy(self, x, y):
        y_pred = self.predict(x)
        score = np.equal(y_pred, y).sum() / x.shape[0]
        return score
        
    def train(self, x, y, batch_size = 64):
        for i in range(0, len(x), batch_size):
            #SGD optimization
            indices = np.random.permutation(len(x))
            x_shuffled = x[indices]
            y_shuffled = y[indices]
            for i in range(0, len(x), batch_size):
                #Get mini-batch
                x_batch = x_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]

                z = np.dot(x_batch, self.w) + self.b
                y_prob = 1 /(1 + np.exp(-z))
                error = y_prob - y_batch
                dev_w = np.dot(x_batch.T, error) / x_batch.shape[0]
                dev_b = np.mean(error, axis=0)
                self.w -= self.lr * dev_w
                self.b -= self.lr * dev_b