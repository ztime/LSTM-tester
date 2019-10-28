from keras.layers import Input, LSTM, ConvLSTM2D, Conv3D, BatchNormalization, RepeatVector, TimeDistributed
from keras.models import Sequential, Model
from keras.optimizers import Adam, RMSprop
from keras import backend as K
import numpy as np
import copy

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
            optimizer='adam',
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
    decoder_1_y_train = np.zeros((total_sequences, sequence_length, img_width, img_height, img_channels))
    decoder_2_y_train = np.zeros((total_sequences, img_width, img_height, img_channels))
    for sequence in range(total_sequences):
        reverse_x_values = copy.copy(x_train[sequence])
        np.flip(reverse_x_values, axis=0)
        decoder_1_y_train[sequence] = reverse_x_values
        decoder_2_y_train[sequence] = y_train[sequence]

    return x_train, [decoder_1_y_train, decoder_2_y_train]

def _build_network(sequence_length, img_width, img_height):
    input_layer = Input(shape=(sequence_length, img_width, img_height, 1))
    encoder = ConvLSTM2D( filters=128, kernel_size=(3,3), padding='same', return_sequences=True)(input_layer)
    # Reconstruct decoder
    # decoder1 = RepeatVector(sequence_length)(encoder)
    decoder1 = ConvLSTM2D(filters=128, kernel_size=(3,3), padding='same', return_sequences=True)(encoder)
    decoder1 = ConvLSTM2D(filters=1, kernel_size=(3,3), padding='same', return_sequences=True)(decoder1)
    # decoder1 = TimeDistributed(Conv3D(filters=1, kernel_size=(3,3,3), padding='same', activation='sigmoid', data_format='channels_last'))(decoder1)
    # decoder1 = Conv3D(filters=1, kernel_size=(3,3,3), padding='same', activation='sigmoid', data_format='channels_last')(decoder1)
    # Predict decoder
    # decoder2 = RepeatVector(sequence_length - 1)(encoder)
    decoder2 = ConvLSTM2D(filters=128, kernel_size=(3,3), padding='same', return_sequences=True)(encoder)
    decoder2 = ConvLSTM2D(filters=1, kernel_size=(3,3), padding='same', return_sequences=False)(decoder2)
    # decoder2 = TimeDistributed(Conv3D(filters=1, kernel_size=(3,3,3), padding='same', activation='sigmoid', data_format='channels_last'))(decoder2)
    # decoder2 = Conv3D(filters=1, kernel_size=(3,3,3), padding='same', activation='sigmoid', data_format='channels_last')(decoder2)
    # together
    model = Model(inputs=input_layer, outputs=[decoder1, decoder2])
    return model

