import numpy as np

class MCLogisticRegression:
    def __init__(self, x, y, x_test, y_test, lr, epochs, n_class, weight=None, intercept=None):
        n_features = x.shape[1]
        self.n_classes = n_class
        if weight is None or intercept is None:
            self.w  = np.random.uniform(-1, 1, (n_features, self.n_classes))
            self.b = np.random.uniform(-1, 1, self.n_classes)
        else:
            self.w = weight
            self.b = intercept
        self.lr = lr
        self.epochs = epochs
        self.x = x
        self.y = y
        self.x_test = x_test
        self.y_test = y_test
        
        self.train_acc = []
        self.test_acc = []
        
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def predict(self, x): 
        z = np.dot(x, self.w) + self.b
        y_prob = self.softmax(z)
        y_pred = np.argmax(y_prob, axis=1)
        return y_pred
    
    def accuracy(self, x, y):
        y_pred = self.predict(x)
        score = np.equal(y_pred, y).sum() / x.shape[0]
        return score
    
    def train(self, eval=False):
        for i in range(self.epochs):
            z = np.dot(self.x, self.w) + self.b
            y_prob = self.softmax(z)
            # One-hot encoding of y
            y_one_hot = np.eye(self.n_classes)[self.y]
            error = y_prob - y_one_hot
            dev_w = np.dot(self.x.T, error) / self.x.shape[0]
            dev_b = np.mean(error, axis=0)
            self.w = self.w - self.lr * dev_w
            self.b = self.b - self.lr * dev_b
            if(eval == True): 
                #self.train_acc.append(self.accuracy(self.x, self.y))
                self.test_acc.append(self.accuracy(self.x_test, self.y_test))