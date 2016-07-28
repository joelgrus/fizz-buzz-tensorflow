import numpy as np
import tensorflow as tf

# The numbers to generate fizz buzz for.
numbers = np.arange(1, 101)

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

class BinaryEncodedData():
    def __init__(self, num_digits=10):
        self.num_digits = num_digits
        self.num_inputs = num_digits

        self.trX = np.array([binary_encode(i, num_digits) for i in range(101, 2 ** num_digits)])
        self.trY = np.array([fizz_buzz_encode(i)          for i in range(101, 2 ** num_digits)])

        self.teX = np.transpose(binary_encode(numbers, num_digits))

        # Our variables. The input has width num_digits, and the output has width 4.
        self.X = tf.placeholder("float", [None, num_digits])
        self.Y = tf.placeholder("float", [None, 4])


def digits(i, num_digits):
    """represents the integer `i` as an array of `num_digits` decimal digits"""
    return np.array([(i // 10 ** d) % 10 for d in range(num_digits)])

def one_hot_digit_encode(d):
    """encodes the digit `d` as a one-hot array of length 10"""
    digits = np.zeros(10)
    digits[d] = 1
    return digits

def one_hot_decimal_encode(i, num_digits):
    """encodes the number `i` as an array of one-hot digit encodings of length
    `10 * num_digits`"""
    return np.concatenate(
        [one_hot_digit_encode(d)
         for d in digits(i, num_digits)])

class DecimalEncodedData():
    def __init__(self, num_digits=3):
        self.num_digits = num_digits
        self.num_inputs = num_digits * 10

        self.X = tf.placeholder("float", [None, 10 * num_digits])
        self.Y = tf.placeholder("float", [None, 4])

        self.trX = np.array([one_hot_decimal_encode(i, num_digits) for i in range(101, 1000)])
        self.trY = np.array([fizz_buzz_encode(i)                   for i in range(101, 1000)])

        self.teX = (np.concatenate([one_hot_decimal_encode(i, num_digits) for i in numbers])
                    .reshape(100, num_digits * 10))

def data_from_args(parsed_args, **kwargs):
    if parsed_args.decimal:
        num_digits = parsed_args.num_digits or kwargs.get("num_digits") or 3
        return DecimalEncodedData(num_digits)
    else:
        num_digits = parsed_args.num_digits or kwargs.get("num_digits") or 10
        return BinaryEncodedData(num_digits)
