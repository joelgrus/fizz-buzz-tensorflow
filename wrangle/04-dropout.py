# since too many hidden neurons seemed to be overfitting, I thought I'd try
# using dropout. turns out it doesn't work here.
# https://en.wikipedia.org/wiki/Dropout_(neural_networks)

import tensorflow as tf
import numpy as np

from command_line_args import arg_parser
from data import data_from_args
from model_helpers import build_model, init_weights, set_seeds

args = arg_parser.parse_args()
data = data_from_args(args)

NUM_HIDDEN = args.num_hidden or 200

set_seeds(123)

# Initialize the weights.
w_h = init_weights([data.num_inputs, NUM_HIDDEN])
w_o = init_weights([NUM_HIDDEN, 4])

b_h = init_weights([NUM_HIDDEN])
b_o = init_weights([4])

def model(X, w_h, w_o, b_h, b_o, keep_prob):
    h = tf.nn.dropout(tf.nn.relu(tf.matmul(X, w_h) + b_h), keep_prob)
    return tf.matmul(h, w_o) + b_o

def py_x(keep_prob):
    return model(data.X, w_h, w_o, b_h, b_o, keep_prob)

keep_prob = args.keep_prob or 0.50

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x(keep_prob=keep_prob), data.Y))
train_op = tf.train.RMSPropOptimizer(learning_rate=0.0003, decay=0.8, momentum=0.4).minimize(cost)

predict_op = tf.argmax(py_x(keep_prob=1.0), 1)

with tf.Session() as sess:
    output = build_model(sess, data, train_op, predict_op, args)
    learned_parameters = sess.run((w_h, w_o, b_h, b_o))
    print(output)
