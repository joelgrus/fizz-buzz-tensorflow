# Linear Regression model
# https://en.wikipedia.org/wiki/Linear_regression

import tensorflow as tf
import numpy as np

from command_line_args import arg_parser
from data import data_from_args
from model_helpers import build_model, init_weights, set_seeds, make_output

set_seeds(123)

args = arg_parser.parse_args()
data = data_from_args(args)

w = init_weights([data.num_inputs, 4])
b = init_weights([4])

def model(X, w, b):
    return tf.matmul(X, w) + b

# Predict y given x using the model.
py_x = model(data.X, w, b)

# We'll train our model by minimizing a cost function.
cost = tf.reduce_mean(tf.pow(py_x - data.Y, 2))
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost)

# And we'll make predictions by choosing the largest output.
predict_op = tf.argmax(py_x, 1)

with tf.Session() as sess:
    output = build_model(sess, data, train_op, predict_op, args)
    learned_parameters = sess.run((w, b))
    print(output)
