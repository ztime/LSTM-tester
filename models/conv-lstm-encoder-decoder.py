from keras.layers import Input, LSTM, ConvLSTM2D, Conv3D, BatchNormalization
from keras.layers import Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam, RMSprop
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import backend as K

from custom_loss import huber_loss
import keras
import numpy as np
import copy

MODEL_OVERRIDES = {
        "sequence_length": 19,
        "batchsize": 1,
        "data_prepare": True,
        }

def get_description():
    desc = ["Recreating the encoder decoder example that gave something"]
    return '\n'.join(desc)

def get_model(sequence_length, img_width, img_height):
    model = _build_network(sequence_length, img_width, img_height)

    model.compile(
            loss=huber_loss,
            # optimizer=Adam(),
            optimizer='rmsprop',
            metrics=[
                'accuracy',
                'mse',
                ]
            )
    return model

def data_prepare(x_train, y_train):
    """
    We are actually preciting sequences here, so we need to adjust this a bit

    Very wierd

    """
    total_sequences, sequence_length, img_width, img_height, img_channels = x_train.shape
    y_train_new = np.zeros((total_sequences, sequence_length, img_width, img_height, img_channels))
    for sequence in range(total_sequences):
        # Shift all frames so that the first frame ends up last
        # i.e [1, 2, 3] - > [2, 3, 1]
        for i in range(sequence_length - 1):
            y_train_new[sequence][i] = copy.copy(x_train[sequence][i + 1])
        # replace the last frame with prediction  [2, 3, 1] -> [2, 3, 4]
        y_train_new[sequence][sequence_length - 1] = y_train[sequence]

    # import matplotlib.pyplot as plt
    # plt.tight_layout()
    # fig1 = plt.figure(1)
    # row, col = 2, 12
    # for i in range(0, row * col):
        # ax = fig1.add_subplot(row, col, i + 1)
        # if i > 11:
            # ax.imshow(y_train_new[0,i % 12][:,:,0], cmap='gray')
        # else:
            # ax.imshow(x_train[0,i % 12][:,:,0], cmap='gray')
    # fig1.show()
    # input()
    # quit()

    return x_train, y_train_new


def _build_network(sequence_length, img_width, img_height):
    seq = Sequential()
    seq.add(ConvLSTM2D(filters=32, kernel_size=(3, 3), input_shape=(None, img_width, img_height, 1), padding='same', return_sequences=True))
    # seq.add(ConvLSTM2D(filters=16, kernel_size=(3, 3), padding='same', return_sequences=False))
    seq.add(ConvLSTM2D(filters=16, kernel_size=(3, 3), padding='same', return_sequences=True))
    seq.add(Conv3D(filters=16, kernel_size=(3,3,1), padding='same', activation='relu', data_format='channels_last', use_bias=True))
    seq.add(Conv3D(filters=8, kernel_size=(3,3,1), padding='same', activation='relu', data_format='channels_last', use_bias=True))
    seq.add(Conv3D(filters=1, kernel_size=(3,3,1), padding='same', activation=None, data_format='channels_last', use_bias=True))
    # seq.add(Conv2D(filters=16, kernel_size=(3,3), padding='same', activation='relu', data_format='channels_last', use_bias=True))
    return seq

