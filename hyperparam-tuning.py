# Hyperparameter Tuning for Neural Network with GridSearchCV

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

# Step 1: Generate synthetic dataset
X, y = make_classification(n_samples=2000, n_features=20, 
                           n_informative=15, n_classes=2, random_state=42)

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Step 2: Build model function (needed for KerasClassifier)
def create_model(neurons=16, activation='relu'):
    model = Sequential()
    model.add(Dense(neurons, input_dim=X.shape[1], activation=activation))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Step 3: Wrap model with KerasClassifier
model = KerasClassifier(build_fn=create_model, verbose=0)

# Step 4: Define hyperparameter grid
param_grid = {
    'neurons': [8, 16, 32],
    'activation': ['relu', 'tanh'],
    'batch_size': [16, 32],
    'epochs': [10, 20]
}

# Step 5: Grid search
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, verbose=1, n_jobs=-1)
grid_result = grid.fit(X, y)

# Step 6: Print results
print("Best Accuracy: {:.2f}% using {}".format(grid_result.best_score_ * 100, grid_result.best_params_))

# Optional: See all results
means = grid_result.cv_results_['mean_test_score']
params = grid_result.cv_results_['params']
for mean, param in zip(means, params):
    print("Accuracy: {:.2f}% with: {}".format(mean * 100, param))
