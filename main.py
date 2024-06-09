import numpy as np
from PIL import Image
from keras import models, layers, utils
import os

class_names = ['paper', 'rock', 'scissors']
img_width = 150
img_height = 150

x_train = np.zeros((2520, img_width, img_height, 3), 'uint8')
x_test = np.zeros((372, img_width, img_height, 3), 'uint8')

label_train = np.zeros(2520)
label_test = np.zeros(372)

# train data
for i in range(len(class_names)):
    path = f'./data/train/{class_names[i]}'
    files = os.listdir(path)

    for j in range(len(files)):
        img = Image.open(os.path.join(path, files[j]))
        img = img.convert('RGB')
        img = img.resize((img_width, img_height))
        img_arr = np.array(img, 'uint8')

        x_train[(i + 1) * j] = img_arr
        label_train[(i + 1) * j] = i

# test data
for i in range(len(class_names)):
    path = f'./data/test/{class_names[i]}'
    files = os.listdir(path)

    for j in range(len(files)):
        img = Image.open(os.path.join(path, files[j]))
        img = img.convert('RGB')
        img = img.resize((img_width, img_height))
        img_arr = np.array(img, 'uint8')

        x_test[(i + 1) * j] = img_arr
        label_test[(i + 1) * j] = i

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

label_train = utils.to_categorical(label_train, num_classes=len(class_names))
label_test = utils.to_categorical(label_test, num_classes=len(class_names))


CNN = models.Sequential()

CNN.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(img_width, img_height,  3)))
CNN.add(layers.MaxPooling2D((2, 2)))

CNN.add(layers.Conv2D(64, (3, 3), activation='relu'))
CNN.add(layers.MaxPooling2D((2, 2)))

CNN.add(layers.Conv2D(128, (3, 3), activation='relu'))
CNN.add(layers.MaxPooling2D((2, 2)))

CNN.add(layers.Conv2D(128, (3, 3), activation='relu'))
CNN.add(layers.MaxPooling2D((2, 2)))

CNN.add(layers.Flatten())
CNN.add(layers.Dropout(0.5))

CNN.add(layers.Dense(512, activation='relu'))

CNN.add(layers.Dense(3, activation='softmax'))

CNN.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

CNN.fit(x_train, label_train, batch_size=32, epochs=10)

CNN.save('./model/CNN.keras')

test_loss, test_acc = CNN.evaluate(x_test, label_test)
print('test_loss:    ', test_loss)
print('test_accuracy:', test_acc)
