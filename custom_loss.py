import numpy as np
import math
from keras import backend as K
import tensorflow as tf

def euclidian_loss(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_true - y_pred)))

class HackySacky:
    def __init__(self, name):
        self.positions = []
        self.x_counter = 0
        self.y_counter = 0
        self.x_bound = 64
        self.y_bound = 64
        self.name = name

        self.file = f"LOGGYLOGLOG-{name}.log"
        self.write_to_file("/////////////////////////////////////////////")

    def write_to_file(self, s):
        with open(self.file, 'a') as f:
            f.write(f"{s}\n")

    def ping(self, value):
        self.write_to_file(f"x_counter: {self.x_counter} value:{value}")
        # self.write_to_file(f"{value[0,0,0]}")
        # self.write_to_file(f"{K.variable(value)}")
        self.x_counter += 1
        return value

def new_c_loss(y_true, y_pred):
    y_t_hacky = HackySacky('y_true')
    y_p_hacky = HackySacky('y_pred')
    y_t = K.get_value(y_true)
    y_t_hacky.write_to_file(y_t)
    # y_t_n = K.zeros_like(y_true)
    # K.map_fn(y_t_hacky.ping, y_true[0])
    # K.map_fn(y_p_hacky.ping, y_pred[0])
    y_true = tf.map_fn(y_t_hacky.ping, y_true)
    y_pred = tf.map_fn(y_p_hacky.ping, y_pred)
    return K.abs(K.sum(y_true) - K.sum(y_pred))

def count_pixel_loss(y_true, y_pred):
    """
    Assumues that y_true and y_pred are images
    """
    # y_true = K.update_sub(K.sum(y_true), K.sum(y_pred))
    # return y_true
    # y_true = K.update_sub(K.sum(y_true), K.sum(y_pred))
    return K.abs(K.sum(y_true) - K.sum(y_pred))

def log_count_pixel_loss(y_true, y_pred):
    return K.log(count_pixel_loss(y_true, y_pred))

def cross_entropy_from_convlstm(y_true, y_pred):
    test = HackySacky('test')
    y_pred = K.constant(y_pred) if not K.is_tensor(y_pred) else y_pred
    y_true = K.cast(y_true, y_pred.dtype)
    # multiplication = (y_true * K.log(y_pred)) + ((1.0 - y_true) * K.log(1.0 - y_pred))
    one = K.constant(1.0)
    small = K.constant(0.1)
    positive = y_true * K.log(y_pred) * small
    negative = (one - y_true) * K.log(one - y_pred) * small
    addition = positive + negative
    negative_sum = - K.sum(addition)
    # multiplication = tf.add(tf.multiply(y_true, tf.log(y_pred)), tf.multiply((1.0 - y_true), tf.log(1.0 - y_pred)))
    return negative_sum

def binary_crossentropy(y_true, y_pred, from_logits=False, label_smoothing=0):
    y_pred = K.constant(y_pred) if not K.is_tensor(y_pred) else y_pred
    y_true = K.cast(y_true, y_pred.dtype)
    return K.mean(
        K.binary_crossentropy(y_true, y_pred, from_logits=from_logits), axis=-1
    )

def huber_and_count_pixel_loss(y_true, y_pred):
    return huber_loss(y_true, y_pred) + count_pixel_loss(y_true, y_pred)

def cross_convlstm_and_count_pixel_loss(y_true, y_pred):
    return cross_entropy_from_convlstm(y_true, y_pred) + count_pixel_loss(y_true, y_pred)

def bin_cross_and_count_pixel_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + count_pixel_loss(y_true, y_pred)

def norm_loss(y_true, y_pred):
    n_y_t = tf.linalg.norm(y_true)
    n_y_p = tf.linalg.norm(y_pred)
    return n_y_t - n_y_p

def combine_count_and_norm_loss(y_true, y_pred):
    return count_pixel_loss(y_true, y_pred) + norm_loss(y_true, y_pred)

def huber_loss(y_true, y_pred, delta=1.0):
    error = y_pred - y_true
    abs_error = K.abs(error)
    quadratic = K.minimum(abs_error, delta)
    linear = abs_error - quadratic
    return 0.5 * K.square(quadratic) + delta * linear

def combine_euclidian_and_pixel_count(y_true, y_pred):
    eu = euclidian_loss(y_true, y_pred)
    cp = count_pixel_loss(y_true, y_pred)
    # return K.update_add(eu, cp)
    return eu + cp
    # return eu

def pixel_wise_distances(frame):
    """
    Counts the distance between all pixels in frame
    """
    pixel_locations = []
    rows, cols = frame.shape
    for i in range(rows):
        for j in range(cols):
            if frame[i][j] == 1:
                pixel_locations.append((i,j))
    # Maximized range from 0,0 -> rows, cols
    max_length = math.ceil(math.sqrt((rows*rows) + (cols*cols)))
    distances = {}
    for pixel_index in range(len(pixel_locations)):
        x_1, x_2 = pixel_locations[pixel_index]
        for other_pixel in range(pixel_index + 1, len(pixel_locations)):
            y_1, y_2 = pixel_locations[other_pixel]
            dist = euclidian_ceiled(x_1, x_2, y_1, y_2)
            if dist not in distances:
                distances[dist] = 0
            distances[dist] += 1
    # lists for the plot
    x = []
    y = []
    # No zero indexed array, so we use + 1
    for i in range(1, max_length + 1):
        x.append(i)
        if i not in distances:
            y.append(0)
        else:
            y.append(distances[i])

    return x, y

def min_value_in_pred(y_true, y_pred):
    return K.min(y_pred)

def min_value_in_true(y_true, y_pred):
    return K.min(y_true)

def max_value_in_true(y_true, y_pred):
    return K.max(y_true)

def max_value_in_pred(y_true, y_pred):
    return K.max(y_pred)

def euclidian(x_1, x_2, y_1, y_2):
    return math.sqrt( (x_1 - y_1)**2 + (x_2 - y_2)**2 )

def euclidian_ceiled(x_1, x_2, y_1, y_2):
    return math.ceil(euclidian(x_1, x_2, y_1, y_2))

def count_pixels(frame):
    """
    Returns a pixel count like this
    (white pixels, black pixels)

    Data is binary so ones and zeros
    """
    rows, cols = frame.shape
    white_pixels = frame.sum()
    return white_pixels, int((rows * cols) - white_pixels)

def hard_activation(x):
    constant_point_five = K.constant(0.5)
    constant_zero = K.constant(0.0)
    constant_one = K.constant(1.0)
    def f1(): return tf.multiply(x, 0)
    def f2(): return tf.multiply(x, 1)
    return tf.cond(tf.less(x,constant_point_five), f1, f2)
