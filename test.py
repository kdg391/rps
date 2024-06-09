import numpy as np
from PIL import Image
from keras import models

class_names = ['paper', 'rock', 'scissors']

CNN: models.Sequential = models.load_model('./CNN.keras')
CNN.summary()

def test(path: str):
    img = Image.open(path)
    img = img.convert('RGB')
    img = img.resize((150, 150))
    img_arr = np.array(img, 'uint8')
    img_arr = img_arr.reshape((1, 150, 150, 3))

    a = CNN.predict(img_arr)
    b = np.argmax(a, axis=1)

    print('예측:', class_names[b[0]], path)

test('./data/validation/rock1.png')
