import numpy as np
import math
from keras import backend as K
import tensorflow as tf

def count_pixel_loss(y_true, y_pred):
    """
    Assumues that y_true and y_pred are images
    """
    y_t_sum = K.sum(y_true)
    y_p_sum = K.sum(y_pred)

    y_true = K.print_tensor(y_true, message=f"shape y_true:{K.shape(y_true)}")
    print(f"K.shape(y_true):{K.shape(y_true)}")
    print(f"K.shape(y_pred):{K.shape(y_pred)}")
    return K.abs(y_t_sum - y_p_sum)

def log_count_pixel_loss(y_true, y_pred):
    return K.log(count_pixel_loss(y_true, y_pred))

def norm_loss(y_true, y_pred):
    n_y_t = tf.linalg.norm(y_true)
    n_y_p = tf.linalg.norm(y_pred)
    return K.abs(n_y_t - n_y_p)

def combine_count_and_norm_loss(y_true, y_pred):
    return K.log(count_pixel_loss(y_true, y_pred)) + norm_loss(y_true, y_pred)

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
