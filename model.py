import numpy as np

class KNN:
    def __init__(self, k=5):
        self.k = k
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        
    def predict(self, X_test):
        predictions = []
        for x in X_test:
            distances = np.sqrt(np.sum((self.X_train - x)**2, axis=1))
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self.y_train[k_indices]
            prediction = np.bincount(k_nearest_labels).argmax()
            predictions.append(prediction)
        return np.array(predictions)

class LinearRegression:
    def __init__(self, lr=0.1, iterations=1000):
        self.lr = lr
        self.iterations = iterations

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.iterations):
            y_predicted = np.dot(X, self.weights) + self.bias
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        y_approx = np.dot(X, self.weights) + self.bias
        return (y_approx >= 0.5).astype(int)

class LogisticRegression:
    def __init__(self, lr=0.1, max_iterations=5000):
        self.lr = lr
        self.max_iterations = max_iterations

    def sigmoid_function(self, z):
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.theta = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.max_iterations):
            z = np.dot(X, self.theta) + self.bias
            h = self.sigmoid_function(z)
            
            gradient_w = np.dot(X.T, (h - y)) / n_samples
            gradient_b = np.sum(h - y) / n_samples
            
            self.theta -= self.lr * gradient_w
            self.bias -= self.lr * gradient_b

    def predict_prob(self, X):
        z = np.dot(X, self.theta) + self.bias
        return self.sigmoid_function(z)
    
    def predict(self, X, threshold=0.5):
        z = np.dot(X, self.theta) + self.bias
        probs = self.sigmoid_function(z)
        return (probs >= threshold).astype(int)