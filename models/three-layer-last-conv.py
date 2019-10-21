from keras.layers import Input, LSTM, ConvLSTM2D, Conv3D, BatchNormalization
from keras.models import Sequential, Model
from keras.optimizers import Adam, RMSprop
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import backend as K

def get_description():
    desc = [" 3 Layer , dropout .5 , last one is conv3d layer"]
    desc.append("binary crossentropy and rmsprop as optimizer")
    return '\n'.join(desc)

def get_model(sequence_length, img_width, img_height):
    model = _build_network(sequence_length, img_width, img_height)
    rmsprop = RMSprop()
    model.compile(loss='binary_crossentropy', optimizer=rmsprop, metrics=['accuracy'])
    return model

def _build_network(sequence_length, img_width, img_height):
    model = Sequential()
    model.add(
            ConvLSTM2D(
                # filters=self.img_height,
                filters=64,
                # kernel_size=(4,4),
                kernel_size=(3,3),
                input_shape=(sequence_length, img_width, img_height, 1),
                padding='same',
                return_sequences=True,
                dropout=0.5,
                recurrent_dropout=0.5,
                )
        )
    model.add(BatchNormalization())
    model.add(
            ConvLSTM2D(
                filters=64,
                # kernel_size=(4,4),
                kernel_size=(3,3),
                padding='same',
                return_sequences=True,
                dropout=0.5,
                recurrent_dropout=0.5,
                )
        )
    model.add(BatchNormalization())
    model.add(
            Conv3D(filters=1,
                kernel_size=(3,3,3),
                activation='sigmoid',
                padding='same',
                data_format='channels_last',
                )
            )
    return model
