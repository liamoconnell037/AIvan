import tensorflow_hub as hub
import tensorflow_text
from tensorflow import keras
import base64
import numpy as np
from bitarray import bitarray  # pip install bitarray

model = keras.models.load_model('./model.keras')
preprocessor = hub.KerasLayer('https://www.kaggle.com/api/v1/models/tensorflow/bert/tensorFlow2/en-cased-preprocess/3/download')
encoder = hub.KerasLayer(
    "https://www.kaggle.com/api/v1/models/tensorflow/bert/tensorFlow2/en-cased-l-12-h-768-a-12/4/download", trainable=True)


def decode(msg, num=0):
    if num >= 5:
        print("hit the limit for number of recursions")
        return msg
    prediction = model.predict(bertsentence([msg]))
    m = np.argmax(prediction[0])
    match m:
        case 0:
            print("base64!")
            return decode(base64.b64decode(msg).decode('ascii'), num+1)
        case 1:
            print("binary!")
            return decode(bitarray(msg).tobytes().decode('ascii'), num+1)
        case 2:
            print("english!")
            return msg


def bertsentence(sentences):
    return encoder(preprocessor(sentences))['pooled_output']

print("single encoded:")
print(decode('RHVyaW5nIHRoZSBzZXZlbnRlZW50aCBzZWFzb24gb2YgQW1lcmljYSdzIEdvdCBUYWxlbnQsIEpvSm8gYW5kIEplc3NhbHluIGZvcm1lZCB0aGUgZ3JvdXAgWE9NRyBQb3AgdGhhdCB3ZXJlIHByZXZpb3VzbHkgZGlzY292ZXJlZCBvbiBTaXdhJ3MgRGFuY2UgUG9wIFJldm9sdXRpb24gd2hvIGF1ZGl0aW9uZWQgYnkgc2luZ2luZyAiQ2FuZHkgSGVhcnRzIi4gVGhlIGp1ZGdlcyB3ZXJlIGltcHJlc3NlZCBhbmQgcHJvbW90ZWQgdGhlbSB0byB0aGUgbmV4dCByb3VuZC4gSm9KbyBhbmQgSmVzc2FseW4gY29uZ3JhdHVsYXRlZCB0aGUgZ3JvdXAgYXMgSm9KbyB0b2xkIHRoZSBqdWRnZXMgdGhhdCBzaGUgaXMgYSBiaWcgZmFuIG9mIEFHVC4='))

print(decode('eWF5eXl5IEkgbG92ZSB3YXRjaGluZyBjaGFvcw=='))
print(decode('01001011 01000101 01010110 01001001 01001110 00100000 01000100 01010101 01001000 01000001 01001110 01000101 01011001 00100000 01000001 01010011 00100000 01000001 00100000 01010111 01001001 01001110 01000100 01001111 01010111 00100000 01010111 01000001 01010011 01001000 01000101 01010010'))
print(decode("Hahaha! I love typing things into this string"))
print(decode('U29tZXRoaW5nIHNvbWV0aGluZyBBSEhISEhISEhISEhISEhIIG1vcmUgdGhpbmdzIHRvIHR5cGUgaW50byBib3ggd29vb29vb29vb29vb29vbyBJIGxvdmUgZW5jb2RpbmcgYW5kIGFsc28gZGVjb2RpbmcgdGhpbmdzIGl0cyByZWFsbHkgZWFzeSE='))

print("\n\nmultiple encoded:")

print(decode('MDEwMDEwMDAgMDExMDAwMDEgMDExMDEwMDAgMDExMDAwMDEgMDExMDEwMDAgMDExMDAwMDEgMDAxMDAwMDEgMDAxMDAwMDAgMDEwMDEwMDEgMDAxMDAwMDAgMDExMDExMDAgMDExMDExMTEgMDExMTAxMTAgMDExMDAxMDEgMDAxMDAwMDAgMDExMTAxMDAgMDExMTEwMDEgMDExMTAwMDAgMDExMDEwMDEgMDExMDExMTAgMDExMDAxMTEgMDAxMDAwMDAgMDExMTAxMDAgMDExMDEwMDAgMDExMDEwMDEgMDExMDExMTAgMDExMDAxMTEgMDExMTAwMTEgMDAxMDAwMDAgMDExMDEwMDEgMDExMDExMTAgMDExMTAxMDAgMDExMDExMTEgMDAxMDAwMDAgMDExMTAxMDAgMDExMDEwMDAgMDExMDEwMDEgMDExMTAwMTEgMDAxMDAwMDAgMDExMTAwMTEgMDExMTAxMDAgMDExMTAwMTAgMDExMDEwMDEgMDExMDExMTAgMDExMDAxMTE='))

print(decode(input('\n\nEnter your own string: ')))
