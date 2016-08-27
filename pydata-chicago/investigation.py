import keras
import numpy as np

model = keras.models.load_model('shallow_25.h5')

numbers = np.arange(1, 1024)
x = binary_encode(numbers, 10).transpose()
y = model.predict(x)

results = list(zip(numbers, y))

results.sort(key=lambda pair: pair[1][-1], reverse=True)

raw = keras.backend.function([model.layers[0].input],
                             [model.layers[2].output])

y = raw([x])[0]

results = list(zip(numbers, y))

results.sort(key=lambda pair: pair[1][-1], reverse=True)
