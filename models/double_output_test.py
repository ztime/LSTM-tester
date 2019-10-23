from keras.layers import Input, LSTM, ConvLSTM2D, Conv3D, BatchNormalization, RepeatVector, TimeDistributed
from keras.models import Sequential, Model
from keras.optimizers import Adam, RMSprop
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
            optimizer='adam',
            metrics=[
                'accuracy',
                'mean_squared_error',
                ]
            )
    return model

def _build_network(sequence_length, img_width, img_height):
    input_layer = Input(shape=(sequence_length, img_width, img_height, 1))
    encoder = ConvLSTM2D( filters=128, kernel_size=(3,3), padding='same', return_sequences=True)(input_layer)
    # Reconstruct decoder
    # decoder1 = RepeatVector(sequence_length)(encoder)
    decoder1 = ConvLSTM2D(filters=128, kernel_size=(3,3), padding='same', return_sequences=True)(encoder)
    decoder1 = TimeDistributed(Conv3D(filters=1, kernel_size=(3,3,3), padding='same', activation='sigmoid', data_format='channels_last'))(decoder1)
    # Predict decoder
    # decoder2 = RepeatVector(sequence_length - 1)(encoder)
    decoder2 = ConvLSTM2D(filters=128, kernel_size=(3,3), padding='same', return_sequences=False)(encoder)
    decoder2 = Conv3D(filters=1, kernel_size=(3,3,3), padding='same', activation='sigmoid', data_format='channels_last')(decoder2)
    # together
    model = Model(inputs=input_layer, outputs=[decoder1, decoder2])
    return model

