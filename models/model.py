import numpy as np
class LogisticRegression:
    def __init__(self, epochs, n_features, lr = 0.3, n_class = 2, weight=None, intercept=None):
        self.n_classes = n_class
        self.lr = lr
        self.epochs = epochs        
        if weight is None or intercept is None:
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
        
    def train(self, x, y):
        for i in range(self.epochs):
            z = np.dot(x, self.w) + self.b
            y_prob = 1 /(1 + np.exp(-z))
            error = y_prob - y
            dev_w = np.dot(x.T, error) / x.shape[0]
            dev_b = np.mean(error, axis=0)
            self.w -= self.lr * dev_w
            self.b -= self.lr * dev_b