from keras.layers import Input, LSTM, ConvLSTM2D, Conv3D, BatchNormalization
from keras.models import Sequential, Model
from keras.optimizers import Adam, RMSprop
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import backend as K
import numpy as np

MODEL_OVERRIDES = {
        "sequence_length": 12,
        "batchsize": 1,
        "data_prepare": True,
        }

def get_description():
    desc = ["4 layers and shorter sequence length so we can use bigger filters:"]
    desc.append("convlstm - 128 filters")
    desc.append("convlstm - 128 filters")
    desc.append("convlstm - 128 filters")
    desc.append("conv3d - 1 filters")
    desc.append(" Overriding sequence length bc memory implications")
    return '\n'.join(desc)

def get_model(sequence_length, img_width, img_height):
    model = _build_network(sequence_length, img_width, img_height)
    model.compile(
            loss='binary_crossentropy',
            optimizer='adadelta',
            metrics=[
                'accuracy',
                'mean_squared_error',
                ]
            )
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
                filters=128,
                kernel_size=(3,3),
                input_shape=(sequence_length, img_width, img_height, 1),
                padding='same',
                return_sequences=True,
                )
        )
    model.add(
            ConvLSTM2D(
                filters=128,
                kernel_size=(3,3),
                padding='same',
                return_sequences=True,
                )
        )
    model.add(
            ConvLSTM2D(
                filters=128,
                kernel_size=(3,3),
                padding='same',
                return_sequences=True,
                )
        )
    model.add(
            ConvLSTM2D(
                filters=1,
                kernel_size=(3,3),
                padding='same',
                return_sequences=False,
                )
        )
    # model.add(
            # Conv3D(filters=1,
                # kernel_size=(3,3,1),
                # activation='sigmoid',
                # padding='same',
                # data_format='channels_last',
                # )
            # )
    return model

