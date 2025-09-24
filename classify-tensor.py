import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Load and preprocess data
X, y = load_iris(return_X_y=True)
y = OneHotEncoder(sparse=False).fit_transform(y.reshape(-1,1))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation="relu"),
    tf.keras.layers.Dense(3, activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train
model.fit(X_train, y_train, epochs=50, verbose=0)

# Evaluate
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print("Test Accuracy:", acc)
