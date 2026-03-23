#Convolutional Neural Network for MNIST dataset 

from tensorflow.keras import layers, models
import tensorflow as tf

#Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

#Normalize the pixel values
(x_train, x_test) = x_train/255.0, x_test/255.0

#Build the model
model = models.Sequential()

#Add convolutional layers
model.add(layers.Flatten(input_shape=(28, 28)))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(10, activation='softmax'))

#Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#Train the model
model.fit(x_train, y_train, epochs=5)

#Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'\nTest accuracy: {test_acc}')

#Make predictions
predictions = model.predict(x_test)