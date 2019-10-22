from keras.layers import Input, LSTM, ConvLSTM2D, Conv3D, BatchNormalization
from keras.models import Sequential, Model
from keras.optimizers import Adam, RMSprop
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import backend as K

def get_description():
    desc = ["4 layers:"]
    desc.append("convlstm - 128 filters")
    desc.append("convlstm - 128 filters")
    desc.append("convlstm - 64 filters")
    desc.append("convlstm - 1 filters")
    desc.append("Using Kullback Leibler divergence as loss function instead")
    desc.append("Also changed the second layer to 128 filters, see if we get overload")
    return '\n'.join(desc)

def get_model(sequence_length, img_width, img_height):
    model = _build_network(sequence_length, img_width, img_height)
    model.compile(
            loss='kullback_leibler_divergence',
            optimizer='adadelta',
            metrics=[
                'accuracy',
                'mean_squared_error',
                # 'mean_absolute_error',
                # 'mean_absolute_percentage_error',
                # 'mean_squared_logarithmic_error',
                # 'squared_hinge',
                # 'hinge',
                # 'logcosh',
                # # 'huber_loss',
                # 'sparse_categorical_crossentropy',
                # 'binary_crossentropy',
                # 'kullback_leibler_divergence',
                # 'poisson',
                # 'cosine_proximity',
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
    model.add(BatchNormalization())
    model.add(
            ConvLSTM2D(
                filters=128,
                kernel_size=(3,3),
                padding='same',
                return_sequences=True,
                )
        )
    model.add(BatchNormalization())
    model.add(
            ConvLSTM2D(
                filters=64,
                kernel_size=(3,3),
                padding='same',
                return_sequences=True,
                )
        )
    model.add(BatchNormalization())
    model.add(
            ConvLSTM2D(
                filters=1,
                kernel_size=(3,3),
                padding='same',
                return_sequences=True,
                )
        )
    return model

