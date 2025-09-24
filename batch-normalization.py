import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0
x_train = x_train.reshape(-1, 28*28)
x_test = x_test.reshape(-1, 28*28)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

def make_model(use_bn=False):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(256, input_shape=(784,), activation="relu"))
    if use_bn:
        model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(128, activation="relu"))
    if use_bn:
        model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(10, activation="softmax"))
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

# Train without BatchNorm
model1 = make_model(use_bn=False)
model1.fit(x_train, y_train, epochs=5, batch_size=128, verbose=0)
acc1 = model1.evaluate(x_test, y_test, verbose=0)[1]

# Train with BatchNorm
model2 = make_model(use_bn=True)
model2.fit(x_train, y_train, epochs=5, batch_size=128, verbose=0)
acc2 = model2.evaluate(x_test, y_test, verbose=0)[1]

print("Accuracy without BatchNorm:", acc1)
print("Accuracy with BatchNorm:", acc2)
