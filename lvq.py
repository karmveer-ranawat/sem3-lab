import numpy as np

# Sample data: 2 classes
X = np.array([[1,1],[2,1],[1,2],[8,8],[9,8],[8,9]])
y = np.array([0,0,0,1,1,1])

# Initialize codebook vectors (1 per class)
codebooks = np.array([[1,1],[9,9]])
labels = np.array([0,1])

lr = 0.1
epochs = 20

for epoch in range(epochs):
    for xi, yi in zip(X, y):
        # Find nearest codebook
        dists = np.linalg.norm(codebooks - xi, axis=1)
        j = np.argmin(dists)

        # Update towards/away
        if labels[j] == yi:
            codebooks[j] += lr * (xi - codebooks[j])
        else:
            codebooks[j] -= lr * (xi - codebooks[j])

print("Final Codebooks:", codebooks)
