from keras.layers import Input, LSTM, ConvLSTM2D, Conv3D, BatchNormalization, Flatten , Reshape
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
    desc = ["Fully connected regular lstm"]
    desc.append(" Overriding sequence length bc memory implications")
    return '\n'.join(desc)

def get_model(sequence_length, img_width, img_height):
    model = _build_network(sequence_length, img_width, img_height)
    model.compile(
            loss='binary_crossentropy',
            optimizer='rmsprop',
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
    input_layer = Input(shape=(sequence_length, img_width, img_height, 1))
    flat_earth = Flatten()(input_layer)
    middle = LSTM(2048, return_sequence=True)(flat_earth)
    top = LSTM(2048, return_sequence=False)(middle)
    output = Reshape((img_width, img_height, 1))(top)

    model = Model(inputs=input_layer, outputs=output)
    return model

