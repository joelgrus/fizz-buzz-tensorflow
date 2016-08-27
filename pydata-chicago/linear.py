from keras.models import Sequential
from keras.layers import Dense, Activation
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

def binary_encode(i, num_digits=10):
    """represents the integer `i` as an array of `num_digits` binary digits"""
    return np.array([i >> d & 1 for d in range(num_digits)])

def decimal_encode(i, num_digits=3):
    digits = [(i // 10 ** d) % 10 for d in range(num_digits)]
    return np.array([1 if i == d else 0 for d in digits for i in range(10)])

model = Sequential([
  Dense(4, input_dim=10),
])

model.compile(optimizer='rmsprop',
              loss='mse',
              metrics=['accuracy'])

num_digits = 10
numbers = np.arange(1, 101)

trX = np.array([binary_encode(i, num_digits) for i in range(101, 2 ** num_digits)])
trY = np.array([fizz_buzz_encode(i)          for i in range(101, 2 ** num_digits)])

teX = np.transpose(binary_encode(numbers, num_digits))

model.fit(trX, trY, nb_epoch=100)

predictions = [fizz_buzz_decode(i+1, y)
               for i, y in enumerate(np.argmax(model.predict(teX), axis=1))]

plot(predictions)
print(ct(predictions))

# decimal

model = Sequential([
  Dense(4, input_dim=30),
])

model.compile(optimizer='rmsprop',
              loss='mse',
              metrics=['accuracy'])

num_digits = 3
numbers = np.arange(1, 101)

trX = np.array([decimal_encode(i, num_digits) for i in range(101, 1000)])
trY = np.array([fizz_buzz_encode(i)          for i in range(101, 1000)])

teX = np.array([decimal_encode(i, num_digits) for i in numbers])

model.fit(trX, trY, nb_epoch=100)

predictions = [fizz_buzz_decode(i+1, y)
               for i, y in enumerate(np.argmax(model.predict(teX), axis=1))]

plot(predictions)
print(ct(predictions))
