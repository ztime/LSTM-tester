import numpy as np
import math
from keras import backend as K
import tensorflow as tf

def euclidian_loss(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_true - y_pred)))

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
    multiplication = y_true * K.log(y_pred) + (1.0 - y_true) * K.log(1.0 - y_pred)
    return - K.sum(multiplication)

def huber_and_count_pixel_loss(y_true, y_pred):
    return huber_loss(y_true, y_pred) + count_pixel_loss(y_true, y_pred)

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
