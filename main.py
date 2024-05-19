
# pip install tensorflow
# pip install tensorflow_hub
# pip install tensorflow_text
# pip install pandas
# pip install scikit-learn
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import numpy as np
import os
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


# class Linear(layers.Layer):
#     def __init__(self, units=768, input_dim=768):
#         super().__init__()
#         self.units = units
#
#     def build(self, input_shape):
#         self.w = self.add_weight(
#             shape=(input_shape[-1], self.units),
#             initializer="random_normal",
#             trainable=True,
#         )
#         self.b = self.add_weight(shape=(self.units,), initializer="random_normal", trainable=True)
#
#     def call(self, inputs):
#         return tf.matmul(inputs, self.w) + self.b
#
#
# class ActivityRegularizationLayer(layers.Layer):
#     def __init__(self, rate=1e-2):
#         super().__init__()
#         self.rate = rate
#
#     def call(self, inputs):
#         self.add_loss(self.rate * tf.reduce_mean(inputs))
#         return inputs
#
#
# class OuterLayer(layers.Layer):
#     def __init__(self):
#         super().__init__()
#         self.activity_reg = ActivityRegularizationLayer(1e-2)
#
#     def call(self, inputs):
#         return self.activity_reg(inputs)
#
#
# class AIvanModel(keras.Model):
#     def __init__(self, num_classes=3):
#         super().__init__()
#         self.classifier = keras.layers.Dense(num_classes)
#
#     def call(self, inputs):
#         x = inputs
#         return self.classifier(x)

df = pd.read_csv("dataset.csv")
x_train, x_test, y_train, y_test = train_test_split(df['Ciphertext'], df['Type'], test_size=0.2, random_state=42)
model = keras.Sequential()
model.add(layers.Dense(3, name='output'))
model(keras.ops.ones(bertsentence(x_train.values).shape))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
print(model.summary())
model.fit(bertsentence(x_train.values), y_train.values, epochs=100)


#model.save(filepath = "./model.keras") # saves the model's training data and stuff like that
