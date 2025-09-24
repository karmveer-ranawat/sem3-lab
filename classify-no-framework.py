import numpy as np

# Dummy dataset (XOR problem)
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

# Initialize weights
np.random.seed(42)
W1 = np.random.randn(2, 2)   # input → hidden
b1 = np.zeros((1, 2))
W2 = np.random.randn(2, 1)   # hidden → output
b2 = np.zeros((1, 1))

# Sigmoid function
def sigmoid(x): return 1 / (1 + np.exp(-x))
def sigmoid_deriv(x): return x * (1 - x)

# Training
lr = 0.1
for epoch in range(10000):
    # Forward pass
    z1 = X.dot(W1) + b1
    a1 = sigmoid(z1)
    z2 = a1.dot(W2) + b2
    a2 = sigmoid(z2)

    # Backpropagation
    error = y - a2
    d2 = error * sigmoid_deriv(a2)
    d1 = d2.dot(W2.T) * sigmoid_deriv(a1)

    # Update weights
    W2 += a1.T.dot(d2) * lr
    b2 += np.sum(d2, axis=0, keepdims=True) * lr
    W1 += X.T.dot(d1) * lr
    b1 += np.sum(d1, axis=0, keepdims=True) * lr

print("Predictions after training:")
print(a2.round())
