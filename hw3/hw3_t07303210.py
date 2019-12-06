from keras.datasets import fashion_mnist
from keras import layers, models, losses
import numpy as np
import matplotlib.pyplot as plt

((trainX, trainY), (testX, testY)) = fashion_mnist.load_data()

label = ["T-shirt/top", "Trouser", "Pullover",
         "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

trainX = trainX.reshape((60000, 28, 28, 1))
testX = testX.reshape(10000, 28, 28, 1)

trainX = trainX / 255.0
testX = testX / 255.0

model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(
    optimizer="adam",
    loss=losses.sparse_categorical_crossentropy,
    metrics=["accuracy"])

fit = model.fit(trainX, trainY, epochs=15, validation_data=(
    testX, testY), batch_size=100)

figure, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 4))

ax1.plot(fit.history['loss'], label="train loss")
ax1.plot(fit.history['val_loss'], label="test loss")
ax1.set_title("Learning Curve")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.legend(loc="upper right")

ax2.plot(fit.history['accuracy'], label="train acc")
ax2.plot(fit.history['val_accuracy'], label="test acc")
ax2.set_title("Accuracy")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.legend(loc="upper right")

figure.savefig("test.png")
plt.close()
