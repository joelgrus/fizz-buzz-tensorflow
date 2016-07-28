# finally, a deep learning model! two hidden layers
# https://en.wikipedia.org/wiki/Deep_learning

import tensorflow as tf
import numpy as np

from command_line_args import arg_parser
from data import data_from_args
from model_helpers import build_model, init_weights, set_seeds

args = arg_parser.parse_args()
data = data_from_args(args)

NUM_HIDDEN1 = args.num_hidden or 100
NUM_HIDDEN2 = args.num_hidden2 or 100

set_seeds(123)

# Initialize the weights.
w_h1 = init_weights([data.num_inputs, NUM_HIDDEN1])
w_h2 = init_weights([NUM_HIDDEN1, NUM_HIDDEN2])
w_o = init_weights([NUM_HIDDEN2, 4])

b_h1 = init_weights([NUM_HIDDEN1])
b_h2 = init_weights([NUM_HIDDEN2])
b_o = init_weights([4])

def model(X, w_h1, w_h2, w_o, b_h1, b_h2, b_o, keep_prob):
    h1 = tf.nn.dropout(tf.nn.relu(tf.matmul(X, w_h1) + b_h1), keep_prob)
    h2 = tf.nn.relu(tf.matmul(h1, w_h2) + b_h2)
    return tf.matmul(h2, w_o) + b_o

def py_x(keep_prob):
    return model(data.X, w_h1, w_h2, w_o, b_h1, b_h2, b_o, keep_prob)

keep_prob = args.keep_prob or 0.50

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x(keep_prob=keep_prob), data.Y))
train_op = tf.train.RMSPropOptimizer(learning_rate=0.0003, decay=0.8, momentum=0.4).minimize(cost)

predict_op = tf.argmax(py_x(keep_prob=1.0), 1)

with tf.Session() as sess:
    output = build_model(sess, data, train_op, predict_op, args, cost=cost)
    learned_parameters = sess.run((w_h1, w_h2, w_o, b_h1, b_h2, b_o))
    print(output)
