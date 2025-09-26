import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# Training data (XOR)
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

# Build model
model = keras.Sequential([
    layers.Dense(2, activation='sigmoid', input_shape=(2,)),  # hidden layer
    layers.Dense(1, activation='sigmoid')                     # output layer
])

# Compile
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train
model.fit(X, y, epochs=500, verbose=0)

# Predictions
print("Predictions:")
print(model.predict(X))
