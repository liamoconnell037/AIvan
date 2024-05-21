import tensorflow_hub as hub
import tensorflow_text
from tensorflow import keras
import base64
from bitarray import bitarray  # pip install bitarray

model = keras.models.load_model('./model.keras')
preprocessor = hub.KerasLayer('https://www.kaggle.com/api/v1/models/tensorflow/bert/tensorFlow2/en-cased-preprocess/3/download')
encoder = hub.KerasLayer(
    "https://www.kaggle.com/api/v1/models/tensorflow/bert/tensorFlow2/en-cased-l-12-h-768-a-12/4/download", trainable=True)


def decode(msg):
    prediction = model.predict(bertsentence([msg]))
    m = 0
    for i in range(1, len(prediction[0])):
        if prediction[0][i] > prediction[0][m]:
            m = i
    match m:
        case 0:
            print("base64!")
            print("\n\n" + base64.b64decode(msg).decode('ascii'))
        case 1:
            print("binary!")
            print("\n\n" + bitarray(msg).tobytes().decode('ascii'))
        case 2:
            print("english!")
            print("\n\n" + msg)


def bertsentence(sentences):
    return encoder(preprocessor(sentences))['pooled_output']


decode('RHVyaW5nIHRoZSBzZXZlbnRlZW50aCBzZWFzb24gb2YgQW1lcmljYSdzIEdvdCBUYWxlbnQsIEpvSm8gYW5kIEplc3NhbHluIGZvcm1lZCB0aGUgZ3JvdXAgWE9NRyBQb3AgdGhhdCB3ZXJlIHByZXZpb3VzbHkgZGlzY292ZXJlZCBvbiBTaXdhJ3MgRGFuY2UgUG9wIFJldm9sdXRpb24gd2hvIGF1ZGl0aW9uZWQgYnkgc2luZ2luZyAiQ2FuZHkgSGVhcnRzIi4gVGhlIGp1ZGdlcyB3ZXJlIGltcHJlc3NlZCBhbmQgcHJvbW90ZWQgdGhlbSB0byB0aGUgbmV4dCByb3VuZC4gSm9KbyBhbmQgSmVzc2FseW4gY29uZ3JhdHVsYXRlZCB0aGUgZ3JvdXAgYXMgSm9KbyB0b2xkIHRoZSBqdWRnZXMgdGhhdCBzaGUgaXMgYSBiaWcgZmFuIG9mIEFHVC4=')
decode('eWF5eXl5IEkgbG92ZSB3YXRjaGluZyBjaGFvcw==')
decode('01001011 01000101 01010110 01001001 01001110 00100000 01000100 01010101 01001000 01000001 01001110 01000101 01011001 00100000 01000001 01010011 00100000 01000001 00100000 01010111 01001001 01001110 01000100 01001111 01010111 00100000 01010111 01000001 01010011 01001000 01000101 01010010')
decode("Hahaha! I love typing things into this string")
decode('U29tZXRoaW5nIHNvbWV0aGluZyBBSEhISEhISEhISEhISEhIIG1vcmUgdGhpbmdzIHRvIHR5cGUgaW50byBib3ggd29vb29vb29vb29vb29vbyBJIGxvdmUgZW5jb2RpbmcgYW5kIGFsc28gZGVjb2RpbmcgdGhpbmdzIGl0cyByZWFsbHkgZWFzeSE=')

decode(input('\n\nEnter your own string: '))
