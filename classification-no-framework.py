import numpy as np

class NeuralNetwork:
    def _init_(self, input_size, hidden_size, output_size):
        # Initialize weights with He initialization
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size)
        self.b2 = np.zeros((1, output_size))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def forward(self, X):
        # Hidden layer with ReLU activation
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = np.maximum(0, self.z1)  # ReLU
        # Output layer with sigmoid activation
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def backward(self, X, y, lr=0.1):
        m = X.shape[0]  # Number of samples

        # Output layer gradient
        dz2 = self.a2 - y.reshape(-1, 1)
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        # Hidden layer gradient (ReLU derivative)
        dz1 = np.dot(dz2, self.W2.T) * (self.z1 > 0)
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        # Update weights
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

    def train(self, X, y, epochs=1000, lr=0.1):
        for epoch in range(epochs):
            # Forward and backward pass
            self.forward(X)
            self.backward(X, y, lr)

            # Print loss every 100 epochs
            if epoch % 100 == 0:
                loss = -np.mean(y * np.log(self.a2 + 1e-8) + (1 - y) * np.log(1 - self.a2 + 1e-8))
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X, threshold=0.5):
        proba = self.forward(X)
        return (proba >= threshold).astype(int)


# 1. Data Preparation (XOR problem)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

# 2. Initialize and train the network
nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)
nn.train(X, y, epochs=1000, lr=0.1)

# 3. Make predictions
predictions = nn.predict(X)
print("\nFinal Predictions:")
for i in range(len(X)):
    print(f"Input: {X[i]}, Predicted: {predictions[i][0]}, Actual: {y[i]}")

# 4. Print learned weights
print("\nLearned Weights:")
print("W1:", nn.W1)
print("b1:", nn.b1)
print("W2:", nn.W2)
print("b2:", nn.b2)
