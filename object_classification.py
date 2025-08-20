import tensorflow as tf
from tensorflow.keras import datasets, models, layers
import numpy as np

# 1. Load Dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# 2. Flatten images & normalize
train_images = train_images.reshape((train_images.shape[0], -1)) / 255.0
test_images = test_images.reshape((test_images.shape[0], -1)) / 255.0

# 3. Define class names
class_names = ['Airplane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# 4. Build simple NN model
model = models.Sequential([
    layers.Dense(512, activation='relu', input_shape=(3072,)),  # Input layer
    layers.Dense(256, activation='relu'),                       # Hidden layer
    layers.Dense(10)                                            # Output layer
])

# 5. Compile model
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# 6. Train model
model.fit(train_images, train_labels, epochs=10, batch_size=64, validation_data=(test_images, test_labels))

# 7. Evaluate
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"Test accuracy: {test_acc:.2f}")

# 8. Prediction example
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)
print(f"Predicted label: {class_names[np.argmax(predictions[0])]} | True label: {class_names[test_labels[0][0]]}")
