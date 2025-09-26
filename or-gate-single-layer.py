import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=10):
        self.weights = np.zeros(input_size + 1)  # +1 for bias
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activation(self, x):
        return np.where(x >= 0, 1, 0)  # step function

    def predict(self, x):
        x = np.insert(x, 0, 1)  # add bias term
        summation = np.dot(self.weights, x)
        return self.activation(summation)

    def fit(self, X, y):
        for _ in range(self.epochs):
            for xi, target in zip(X, y):
                xi = np.insert(xi, 0, 1)  # add bias
                summation = np.dot(self.weights, xi)
                y_pred = self.activation(summation)
                update = self.learning_rate * (target - y_pred)
                self.weights += update * xi

# Example: Learning the OR logic gate
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,1])

p = Perceptron(input_size=2, learning_rate=0.1, epochs=10)
p.fit(X, y)

# Test predictions
for sample in X:
    print(sample, "->", p.predict(sample))
