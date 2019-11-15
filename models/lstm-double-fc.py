from keras.layers import Input, LSTM, Flatten, Reshape
from keras.models import Sequential, Model
from keras.optimizers import Adam, RMSprop
from keras import backend as K
import numpy as np
import copy

MODEL_OVERRIDES = {
        "sequence_length": 10,
        "batchsize": 1,
        "data_prepare": True,
        }

def get_description():
    desc = ["From the 'Unsupervised learning of sequences lstm' paper"]
    desc.append("Training on the mnist digits")
    desc.append("")
    desc.append("")
    desc.append("")
    desc.append("")
    return '\n'.join(desc)

def get_model(sequence_length, img_width, img_height):
    model = _build_network(sequence_length, img_width, img_height)
    rms = RMSprop(learning_rate=1.e-3, rho=0.9)
    model.compile(
            loss='binary_crossentropy',
            optimizer=rms,
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
    # Input shaping
    input_layer = Input(shape=(sequence_length, img_width, img_height, 1))
    # flattend_layer = Flatten()(input_layer)
    flattend_layer = Reshape((sequence_length, img_width * img_height))(input_layer)
    # Encoder layers
    encoder_first_layer = LSTM(2048, return_sequences=True)(flattend_layer)
    encoder_second_layer, enc_2_h_state, enc_2_c_state = LSTM(2048, return_sequences=True, return_state=True)(encoder_first_layer)
    # Decoder number 1 - Is set to reconstruct the input but in reverse order
    # decoder_1_lstm = LSTM(2048, return_sequences=True)(encoder_second_layer)
    # decoder_1_lstm.set_weights(encoder_second_layer.get_weights())
    # Should i care about sequence length here?????
    # decoder_1_output_layer = Reshape((sequence_length, img_width, img_height, 1))(decoder_1_lstm)
    # decoder_1_output_layer = Reshape((img_width, img_height, 1))(decoder_1_lstm)
    decoder_1_output_layer = Reshape((img_width, img_height))(encoder_second_layer)
    # Decoder number 2 - Is set to predict the future 10 frames
    # decoder_2_lstm = LSTM(2048, return_sequences=True)(encoder_second_layer)
    # decoder_2_lstm.set_weights(encoder_second_layer.get_weights())
    # Should i care about sequence length here?????
    # decoder_2_output_layer = Reshape((sequence_length, img_width, img_height, 1))(decoder_2_lstm)
    # decoder_2_output_layer = Reshape((img_width, img_height, 1))(decoder_2_lstm)
    decoder_2_output_layer = Reshape((img_width, img_height))(encoder_second_layer)

    model = Model(inputs=input_layer, outputs=[decoder_1_output_layer, decoder_2_output_layer])
    return model

