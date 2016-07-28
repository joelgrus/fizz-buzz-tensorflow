import numpy as np
import tensorflow as tf

from data import fizz_buzz_encode, fizz_buzz_decode

def set_seeds(seed):
    tf.set_random_seed(seed)
    np.random.seed(seed)

# The numbers to generate fizz buzz for.
numbers = np.arange(1, 101)

def init_weights(shape):
    """returns a tensorflow variable with random weights"""
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def crosstab(predicted):
    counts = np.zeros((4, 4))
    for i, prediction in zip(numbers, predicted):
        actual = np.argmax(fizz_buzz_encode(i))
        counts[prediction][actual] += 1
    return counts

def make_output(sess, data, predict_op):
    teY = sess.run(predict_op, feed_dict={data.X: data.teX})
    print(crosstab(teY))
    output = np.vectorize(fizz_buzz_decode)(numbers, teY)
    return output

def build_model(sess, data, train_op, predict_op, args, cost=None,
                **kwargs):
    sess.run(tf.initialize_all_variables())

    batch_size = args.batch_size or kwargs.get("batch_size") or 128
    num_epochs = args.num_epochs or kwargs.get("num_epochs") or 10000

    for epoch in range(num_epochs):
        # Shuffle the data before each training iteration.
        p = np.random.permutation(range(len(data.trX)))
        trX, trY = data.trX[p], data.trY[p]

        # Train in batches of 128 inputs.
        for start in range(0, len(trX), batch_size):
            end = start + batch_size
            sess.run(train_op, feed_dict={data.X: trX[start:end], data.Y: trY[start:end]})

        if epoch % 100 == 0:
            # print the current accuracy on the training data.
            print(epoch, np.mean(np.argmax(trY, axis=1) ==
                                 sess.run(predict_op, feed_dict={data.X: trX, data.Y: trY})),
                         sess.run(cost, feed_dict={data.X: trX, data.Y: trY}) if cost is not None else "")

        if epoch % 1000 == 0: print(make_output(sess, data, predict_op))

    return make_output(sess, data, predict_op)
