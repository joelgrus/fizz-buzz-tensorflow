from keras.models import Sequential
from keras.layers import Dense, Activation
from plots import *
import numpy as np

def fizz_buzz_encode(i):
    """encodes the desired fizz-buzz output as a one-hot array of length 4:
    [number, "fizz", "buzz", "fizzbuzz"]"""
    if   i % 15 == 0: return np.array([0, 0, 0, 1])
    elif i % 5  == 0: return np.array([0, 0, 1, 0])
    elif i % 3  == 0: return np.array([0, 1, 0, 0])
    else:             return np.array([1, 0, 0, 0])

def fizz_buzz_decode(i, prediction):
    """decodes a prediction {0, 1, 2, 3} into the corresponding output"""
    return [str(i), "fizz", "buzz", "fizzbuzz"][prediction]

def binary_encode(i, num_digits):
    """represents the integer `i` as an array of `num_digits` binary digits"""
    return np.array([i >> d & 1 for d in range(num_digits)])

num_hidden = 200

model = Sequential([
  Dense(num_hidden, input_dim=10),
  Activation('relu'),
  Dense(4),
  Activation('softmax')
])

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

num_digits = 10
numbers = np.arange(1, 101)

trX = np.array([binary_encode(i, num_digits) for i in range(101, 2 ** num_digits)])
trY = np.array([fizz_buzz_encode(i)          for i in range(101, 2 ** num_digits)])

teX = np.transpose(binary_encode(numbers, num_digits))

model.fit(trX, trY, nb_epoch=1000)
model.save("shallow_" + str(num_hidden) + ".h5")

predictions = [fizz_buzz_decode(i+1, y) for i, y in enumerate(np.argmax(model.predict(teX), axis=1))]

plot(predictions)
print(ct(predictions))
print(sum(val
          for i, row in enumerate(ct(predictions))
          for j, val in enumerate(row)
          if i == j))
