import ssl
import tensorflow as tf
import tensorflow.keras.layers as layers
ssl._create_default_https_context = ssl._create_unverified_context

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=[]):
        if logs.get('accuracy') > 0.95:
            print("\n정확도가 95%에 도달하여 훈련을 멈춤니다!")
            self.model.stop_training = True

callback = myCallback()
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# grayscale to rgb
training_images = training_images.reshape(60000, 28, 28, 1)
training_images = training_images / 255.0
test_images = test_images.reshape(10000, 28, 28, 1)
test_images = test_images / 255.0

# apply conv2d + max pooling
model = tf.keras.models.Sequential([
    layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPool2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPool2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation=tf.nn.relu),
    layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=50)
model.evaluate(test_images, test_labels)

classifications = model.predict(test_images)
print(classifications[0])
print(test_labels[0])
