# the multilayer perceptron where I included bias

import tensorflow as tf
import numpy as np

from command_line_args import arg_parser
from data import data_from_args
from model_helpers import build_model, init_weights, set_seeds

args = arg_parser.parse_args()
data = data_from_args(args)

NUM_HIDDEN = args.num_hidden or 100

set_seeds(123)

# Initialize the weights.
w_h = init_weights([data.num_inputs, NUM_HIDDEN])
w_o = init_weights([NUM_HIDDEN, 4])

b_h = init_weights([NUM_HIDDEN])
b_o = init_weights([4])

def model(X, w_h, w_o, b_h, b_o):
    h = tf.nn.relu(tf.matmul(X, w_h) + b_h)
    return tf.matmul(h, w_o) + b_o

py_x = model(data.X, w_h, w_o, b_h, b_o)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, data.Y))
train_op = tf.train.RMSPropOptimizer(learning_rate=0.0008, decay=0.8, momentum=0.4).minimize(cost)

predict_op = tf.argmax(py_x, 1)

with tf.Session() as sess:
    output = build_model(sess, data, train_op, predict_op, args, cost=cost)
    learned_parameters = sess.run((w_h, w_o, b_h, b_o))
    print(output)
