# python version 3.10
# pip install tensorflow
# pip install tensorflow_hub
# pip install tensorflow_text
# pip install pandas
# pip install scikit-learn
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
from tensorflow import keras
from keras import layers
import pandas as pd
from sklearn.model_selection import train_test_split
# import bert models

preprocessor = hub.KerasLayer('https://www.kaggle.com/api/v1/models/tensorflow/bert/tensorFlow2/en-cased-preprocess/3/download')
encoder = hub.KerasLayer(
    "https://www.kaggle.com/api/v1/models/tensorflow/bert/tensorFlow2/en-cased-l-12-h-768-a-12/4/download", trainable=True)


def bertsentence(sentences):
    return encoder(preprocessor(sentences))['pooled_output']


df = pd.read_csv("dataset.csv")
x_train, x_test, y_train, y_test = train_test_split(df['Ciphertext'], df['Type'], test_size=0.2, random_state=42)
x_train = bertsentence(x_train.values)
x_test = bertsentence(x_test.values)
y_train = tf.convert_to_tensor(y_train.values)
y_test = tf.convert_to_tensor(y_test.values)
model = keras.Sequential()
model.add(layers.Dense(75, name='first', activation='relu'))
model.add(layers.Dense(150, name='second', activation='relu'))
model.add(layers.Dense(3, name='output', activation='softmax'))  # 3 because 3 output options (binary, base64, english)
model(keras.ops.ones(x_train.shape))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=25)
