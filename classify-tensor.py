import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# 1. Data Preparation (XOR problem)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([0, 1, 1, 0], dtype=np.float32)

# 2. Define the model
model = Sequential([
    Dense(4, activation='relu', input_shape=(2,)),  # Hidden layer with 4 neurons
    Dense(1, activation='sigmoid')                  # Output layer
])

# 3. Compile the model
model.compile(optimizer=Adam(learning_rate=0.1),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 4. Train the model
print("Training progress:")
history = model.fit(X, y, epochs=1000, verbose=0)

# Print training history every 100 epochs
for i in range(0, 1000, 100):
    loss, acc = history.history['loss'][i], history.history['accuracy'][i]
    print(f"Epoch {i}: loss = {loss:.4f}, accuracy = {acc:.4f}")

# 5. Make predictions
predictions = model.predict(X)
predicted_classes = (predictions > 0.5).astype(int)

# 6. Display results
print("\nFinal Predictions:")
for i in range(len(X)):
    print(f"Input: {X[i]}, Predicted: {predicted_classes[i][0]} "
          f"(prob: {predictions[i][0]:.4f}), Actual: {y[i]}")

# 7. Print model summary
print("\nModel architecture:")
model.summary()

# 8. Print learned weights
print("\nLearned weights:")
for i, layer in enumerate(model.layers):
    weights, biases = layer.get_weights()
    print(f"Layer {i+1} weights:\n{weights}")
    print(f"Layer {i+1} biases:\n{biases}")
