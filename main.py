
# pip install tensorflow
# pip install tensorflow_hub
# pip install tensorflow_text

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import numpy as np
from tensorflow import keras
from keras import layers

# import bert models
preprocessor = hub.KerasLayer('https://www.kaggle.com/api/v1/models/tensorflow/bert/tensorFlow2/en-cased-preprocess/3/download')
encoder = hub.KerasLayer(
    "https://www.kaggle.com/api/v1/models/tensorflow/bert/tensorFlow2/en-cased-l-12-h-768-a-12/4/download", trainable=True)

# TODO: put the data into tf.data.Dataset objects for easy use with tensorflow
text_test = tf.convert_to_tensor(np.array(["something something", "something else as well"]))
encoder_inputs = preprocessor(text_test)
# encoder inputs keys : input_mask, input_type_ids, input_word_ids

# NUMBEEEERSRSSSSSS
outputs = encoder(encoder_inputs)
# outputs keys : default, encoder_outputs, pooled_output

# numbers without math
pooled_output = outputs["pooled_output"]


class Linear(layers.Layer):
    def __init__(self, units=768, input_dim=768):
        super().__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(shape=(self.units,), initializer="random_normal", trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


class ActivityRegularizationLayer(layers.Layer):
    def __init__(self, rate=1e-2):
        super().__init__()
        self.rate = rate

    def call(self, inputs):
        self.add_loss(self.rate * tf.reduce_mean(inputs))
        return inputs


class OuterLayer(layers.Layer):
    def __init__(self):
        super().__init__()
        self.activity_reg = ActivityRegularizationLayer(1e-2)

    def call(self, inputs):
        return self.activity_reg(inputs)


class AIvanModel(keras.Model):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.layer1 = Linear()
        self.layer2 = OuterLayer()
        self.classifier = keras.layers.Dense(num_classes)

    def call(self, inputs):
        x = self.layer1(inputs)
        x = self.layer2(x)
        return self.classifier(x)


model = AIvanModel(1000)
# yay numbers had the math on them
print(model(pooled_output))

# dataset = ???
# model.fit(dataset, epochs=10)
model.save(filepath = "./model.keras") # saves the model's training data and stuff like that
