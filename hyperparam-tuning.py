import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten

# XOR input and expected output
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
expected_output = np.array([[0], [1], [1], [0]])

# Set random seed for reproducibility
tf.random.set_seed(42)

# Define the model
model_xor = Sequential()
model_xor.add(Flatten(input_shape=(2,)))  # Input layer
model_xor.add(Dense(4, activation='relu'))  # Hidden layer with 4 neurons
model_xor.add(Dense(1, activation='sigmoid'))  # Output layer

# Compile the model
model_xor.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    metrics=['accuracy']
)

# Train the model
model_xor.fit(inputs, expected_output, epochs=50, batch_size=2, verbose=1)

# Test predictions
print("Prediction for [0,1]:", np.round(model_xor.predict(np.array([[0, 1]]))))
print("Prediction for [1,1]:", np.round(model_xor.predict(np.array([[1, 1]]))))
