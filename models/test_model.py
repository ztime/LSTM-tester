from keras.layers import Input, LSTM, ConvLSTM2D, Conv3D, BatchNormalization
from keras.models import Sequential, Model
from keras.optimizers import Adam, RMSprop
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import backend as K
import numpy as np

from custom_loss import *

MODEL_OVERRIDES = {
        "data_prepare": True,
        }

def get_description():
    desc = ["Two layers, just to test custom loss"]
    return '\n'.join(desc)

def get_model(sequence_length, img_width, img_height):
    model = _build_network(sequence_length, img_width, img_height)
    # model.compile(loss=euclidian_loss, optimizer='adadelta', metrics=[count_pixel_loss, log_count_pixel_loss, norm_loss])
    model.compile(loss=new_c_loss, optimizer='adadelta', metrics=['accuracy'])
    # model.compile(loss=combine_count_and_norm_loss, optimizer='adadelta', metrics=[count_pixel_loss, log_count_pixel_loss, norm_loss])
    return model

def data_prepare(x_train, y_train):
    """
    For this model we have two outputs so we need two different
    outputs for the loss function to use.

    The first output is for decoder 1 which is the reverse of the input sequence
    ALL of which has been seen in the training data
    Second output is for decoder 2 which is the next frame in the sequence
    which is not in the training data

    x_train will remain the same for the data, so no need to touch that
    """
    total_sequences, sequence_length, img_width, img_height, img_channels = x_train.shape
    y_train = np.reshape(y_train, (total_sequences, img_width, img_height, img_channels))

    return x_train, y_train


def _build_network(sequence_length, img_width, img_height):
    model = Sequential()
    model.add(
            ConvLSTM2D(
                filters=2,
                kernel_size=(3,3),
                input_shape=(sequence_length, img_width, img_height, 1),
                padding='same',
                return_sequences=True,
                dropout=0.5,
                recurrent_dropout=0.5,
                )
        )
    model.add(
            ConvLSTM2D(
                filters=1,
                kernel_size=(3,3),
                padding='same',
                return_sequences=False,
                dropout=0.5,
                recurrent_dropout=0.5,
                )
        )
    return model
