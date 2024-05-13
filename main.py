
# pip install tensorflow
# pip install tensorflow_hub
# pip install tensorflow_text

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

# import bert models
preprocessor = hub.KerasLayer('https://kaggle.com/models/tensorflow/bert/TensorFlow2/en-uncased-preprocess/3')
encoder = hub.KerasLayer(
    "https://www.kaggle.com/models/tensorflow/bert/TensorFlow2/bert-en-uncased-l-10-h-128-a-2/2", trainable=True)

# TODO: how are we gonna get the data in this
text_test = tf.keras.layers.Input(shape=(), dtype=tf.string)
encoder_inputs = preprocessor(text_test)
# encoder inputs keys : input_mask, input_type_ids, input_word_ids

outputs = encoder(encoder_inputs)
# outputs keys : default, encoder_outputs, pooled_output

pooled_output = outputs["pooled_output"]

embedding_model = tf.keras.Model(text_test, pooled_output)
sentences = tf.constant(["123"])

# bunch of numbers that can be used with ai
print(embedding_model(sentences))