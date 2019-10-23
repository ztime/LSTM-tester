from keras.layers import Input, LSTM, ConvLSTM2D, Conv3D, BatchNormalization
from keras.models import Sequential, Model
from keras.optimizers import Adam, RMSprop
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import backend as K

MODEL_OVERRIDES = {
        "sequence_length": 12,
        "batchsize": 1,
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
            Conv3D(filters=1,
                kernel_size=(3,3,3),
                activation='sigmoid',
                padding='same',
                data_format='channels_last',
                )
            )
    return model

