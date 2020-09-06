import cv2
import matplotlib.pyplot as plt
import numpy as np
import requests
from PIL import Image
from keras.datasets import mnist
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical


def le_net_model():
    """
    Creates a model using a customized version of the LeNet-5 Convolutional Neural Network architecture.
    :return: a new sequential model using the LeNet-5 CNN structure.
    """
    new_model = Sequential()
    new_model.add(Conv2D(filters=30, kernel_size=(5, 5), input_shape=(28, 28, 1), activation='relu'))
    new_model.add(MaxPooling2D(pool_size=(2, 2)))
    new_model.add(Conv2D(filters=15, kernel_size=(3, 3), activation='relu'))
    new_model.add(MaxPooling2D(pool_size=(2, 2)))
    new_model.add(Flatten())
    new_model.add(Dense(units=500, activation='relu'))
    new_model.add(Dropout(rate=0.5))
    new_model.add(Dense(units=number_of_classes, activation='softmax'))
    new_model.compile(Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
    return new_model


def show_loss_graph():
    """
    Displays a graph about the validation loss of the model.
    """
    plt.figure(figsize=(12, 4))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['Loss', 'Validation Loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.show()


def show_accuracy_graph():
    """
    Displays a graph about the validation accuracy of the model.
    """
    plt.figure(figsize=(12, 4))
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.legend(['Accuracy', 'Validation Accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.show()


def evaluate_model():
    """
    Evaluates the model to check its score against the testing dataset.
    """
    score = model.evaluate(x=X_testing, y=y_testing, verbose=0)
    print('Test Score: ', score[0])
    print('Test Accuracy: ', score[1])


np.random.seed(0)

number_of_classes = 10  # Numbers ranging from 0 to 9

# Loads data images for the training and testing sets.
(X_training, y_training), (X_testing, y_testing) = mnist.load_data()

# print(X_training.shape)  # 60000 images, made from 28 x 28 pixels.
# print(X_testing.shape)  # 10000 images, made from 28 x 28 pixels.

# Verifies if the complex dataset was imported correctly.
# If not, shows the error message described below.
assert (X_training.shape[0] == y_training.shape[0]), "The number of images is not equal to the number of labels."
assert (X_testing.shape[0] == y_testing.shape[0]), "The number of images is not equal to the number of labels."
assert (X_training.shape[1:] == (28, 28)), "The dimensions of the images are not 28 x 28."
assert (X_testing.shape[1:] == (28, 28)), "The dimensions of the images are not 28 x 28."

number_of_pixels = 784
X_training = X_training.reshape(60000, 28, 28, 1)
X_testing = X_testing.reshape(10000, 28, 28, 1)

# Hot encode to remove any relationship between the labels of the classes, converting them into independent labels.
y_training = to_categorical(y=y_training, num_classes=number_of_classes)
y_testing = to_categorical(y=y_testing, num_classes=number_of_classes)

# Normalizing the data.
# As the images are in grayscale, and each pixel corresponds to a value ranging from 0 to 255,
# we divide them by to 255 to get values ranging from 0 to to the maximum value of 1. Due to this, we decrease the
# amount of variation in our sample. Thus, making the neural network better deal with the input data and learn faster
# and more accurately.
X_training = X_training / 255
X_testing = X_testing / 255

model = le_net_model()
history = model.fit(x=X_training, y=y_training, epochs=10, validation_split=0.1, batch_size=400, verbose=1, shuffle=1)

# show_loss_graph()
# show_accuracy_graph()

# evaluate_dataset()

# Retrieves the image that will be used to validate the neural network.
url = 'https://www.researchgate.net/profile/Jose_Sempere/publication/221258631/figure/fig1/AS:305526891139075@1449854695342/Handwritten-digit-2.png'
response = requests.get(url=url, stream=True)
img = Image.open(response.raw)

# Transforms the current image as an array.
img_array = np.asarray(img)
# Resizing the image to match the dimensions with the neural network was trained.
resized_image = cv2.resize(src=img_array, dsize=(28, 28))
# Converting the image from RGB to GRAYSCALE to match with the neural network color mapping.
grayscale_image = cv2.cvtColor(src=resized_image, code=cv2.COLOR_BGR2GRAY)
# The current GRAYSCALE image is with a white background and black digit. We need to change the image to be with
# black background and white digit to match the neural network training image samples.
corrected_image = cv2.bitwise_not(src=grayscale_image)

corrected_image = corrected_image / 255  # Normalizing the image as the ones before.
corrected_image = corrected_image.reshape(1, 28, 28, 1)  # Reshaping the image array as before.

prediction = np.argmax(model.predict(corrected_image), axis=-1)
print("Predicted digit is: ", str(prediction))
