import pandas as pd
import numpy as np
import cv2
import os
import tensorflow as tf

from keras.utils import to_categorical

# Carrega os dados de emoção do FERPlus dataset
data = pd.read_csv('fer2013new.csv')

emotions = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt', 'unknown', 'NF']

emotion_labels = []
for i in range(len(data)):
    emotion_values = data.loc[i, emotions].values.tolist()
    max_index = emotion_values.index(max(emotion_values))
    emotion_label = emotions[max_index]
    emotion_labels.append(emotion_label)

# Percorre o dataset e insere a coluna pixels com os valores requiridos
pixels = []

for i in range(len(data)):
    pixels_str = ' '.join([str(pix) for pix in data.loc[i, 'pixels']])
    pixels.append(pixels_str)

data['pixels'] = pixels

# Separe as emoções e as imagens em listas
emotion = emotion_labels
if 'pixels' in data.columns:
    print("A coluna 'pixels' está presente no DataFrame 'data'.")
    pixels = data['pixels'].values.tolist()
else:
    print("A coluna 'pixels' não está presente no DataFrame 'data'.")

# Cria uma lista vazia para armazenar as imagens pré-processadas
images = []

# Pré-processamento de imagem
for pixel_sequence in pixels:
    pixel_list = [int(pixel) for pixel in pixel_sequence.split()]
    image = np.array(pixel_list, dtype='uint8').reshape(48, 48)
    image = cv2.equalizeHist(image)
    images.append(image)

# Converta as listas de imagens e emoções para arrays numpy
images = np.array(images)
emotion = np.array(emotion)

# Normalização das imagens
images = images.reshape(images.shape[0], 48, 48, 1)
images = images.astype('float32') / 255.0

# Divisão dos dados em treinamento, validação e teste
train_images = images[:25000]
train_emotion = emotion[:25000]
val_images = images[25000:30000]
val_emotion = emotion[25000:30000]
test_images = images[30000:]
test_emotion = emotion[30000:]

# Defina o número de classes de emoção
num_classes = 8

# Crie um modelo sequencial
model = tf.keras.models.Sequential()

# Adicione camadas convolucionais
model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Flatten())

# Adicione camadas totalmente conectadas
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

# Compile o modelo
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Treina o modelo
model.fit(train_images, to_categorical(train_emotion, num_classes=num_classes), epochs=10, validation_data=(val_images, to_categorical(val_emotion, num_classes=num_classes)))

# Passa as informações do treinamento
test_loss, test_accuracy = model.evaluate(test_images, to_categorical(test_emotion, num_classes=num_classes), verbose=0)
print('Test loss:', test_loss)
print('Test accuracy:', test_accuracy)
