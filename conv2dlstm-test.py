from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import Callback
import numpy as np
import pylab as plt
import os
import copy

# We create a layer which take as input movies of shape
# (n_frames, width, height, channels) and returns a movie
# of identical shape.

seq = Sequential()
seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
    input_shape=(None, 40, 40, 1),
    padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
    padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
    padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
    padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(Conv3D(filters=1, kernel_size=(3, 3, 3),
    activation='sigmoid',
    padding='same', data_format='channels_last'))
# seq.compile(loss='binary_crossentropy', optimizer='adadelta')
class BatchMetrics(Callback):

    def __init__(self, file_path):
        self.file_path = file_path
        super().__init__()

    def on_batch_end(self, batch, loss):
        # We remove 'batch' and 'size' since
        # they are not relevant
        loss_dict = copy.copy(loss)
        loss_dict.pop('batch')
        loss_dict.pop('size')
        if not os.path.isfile(self.file_path):
            # It is our first pass! Create the file
            # and add headers
            with open(self.file_path, 'w') as f:
                f.write('batch\t')
                for key in loss_dict:
                    f.write(f'{key}\t')
                f.write('\n')
        # Now we can print every line
        with open(self.file_path, 'a') as f:
            f.write(f'{batch}\t')
            for key in loss_dict:
                f.write(f'{loss_dict[key]}\t')
            f.write('\n')
        # Done

batch_file = os.path.join('Metrics-each-batch-lstmconv2d.log')
batch_metrics = BatchMetrics(batch_file)

seq.compile(
        loss='binary_crossentropy',
        optimizer='adadelta',
        metrics=[
            'accuracy',
            'mean_squared_error',
            'mean_absolute_error',
            'mean_absolute_percentage_error',
            'mean_squared_logarithmic_error',
            'squared_hinge',
            'hinge',
            'logcosh',
            'huber_loss',
            'sparse_categorical_crossentropy',
            'binary_crossentropy',
            'kullback_leibler_divergence',
            'poisson',
            'cosine_proximity',
            ]
        )


# Artificial data generation:
# Generate movies with 3 to 7 moving squares inside.
# The squares are of shape 1x1 or 2x2 pixels,
# which move linearly over time.
# For convenience we first create movies with bigger width and height (80x80)
# and at the end we select a 40x40 window.

def generate_movies(n_samples=1200, n_frames=15):
    row = 80
    col = 80
    # This contains all the data, split it up later into noisy and shifted
    buffer = np.zeros((n_samples, n_frames + 1, row, col, 1), dtype=np.float)

    for sample_index in range(n_samples):
        # Add 3 to 7 moving squares
        n = np.random.randint(3, 8)

        for box_index in range(n):
            # Initial position
            xstart = np.random.randint(20, 60)
            ystart = np.random.randint(20, 60)
            # Direction of motion
            directionx = np.random.randint(0, 3) - 1
            directiony = np.random.randint(0, 3) - 1

            # Size of the square
            w = np.random.randint(2, 4)

            for t in range(n_frames + 1):
                x_shift = xstart + directionx * t
                y_shift = ystart + directiony * t
                buffer[sample_index, t, x_shift - w: x_shift + w, y_shift - w: y_shift + w, 0] += 1

                # Make it more robust by adding noise.
                # The idea is that if during inference,
                # the value of the pixel is not exactly one,
                # we need to train the network to be robust and still
                # consider it as a pixel belonging to a square.
                if np.random.randint(0, 2):
                    noise_f = (-1)**np.random.randint(0, 2)
                    buffer[sample_index, t, x_shift - w - 1: x_shift + w + 1, y_shift - w - 1: y_shift + w + 1, 0] += noise_f * 0.1

    noisy_movies = np.zeros((n_samples, n_frames, row, col, 1), dtype=np.float)
    shifted_movies = np.zeros((n_samples, n_frames, row, col, 1), dtype=np.float)
    for sample_index in range(n_samples):
        noisy_movies[sample_index] = buffer[sample_index,:n_frames,::,::]
        shifted_movies[sample_index] = buffer[sample_index,1:n_frames+1,::,::]
    # Cut to a 40x40 window
    noisy_movies = noisy_movies[::, ::, 20:60, 20:60, ::]
    shifted_movies = shifted_movies[::, ::, 20:60, 20:60, ::]
    noisy_movies[noisy_movies >= 1] = 1
    shifted_movies[shifted_movies >= 1] = 1
    return noisy_movies, shifted_movies

# Train the network
print("Generate movies...")
noisy_movies, shifted_movies = generate_movies(n_samples=1200)
print("Training")
try:
    seq.fit(
            noisy_movies[:1000],
            shifted_movies[:1000],
            batch_size=10,
            epochs=300,
            validation_split=0.05,
            callbacks=[batch_metrics])
    # seq.fit(noisy_movies[:2], shifted_movies[:2], batch_size=1, epochs=1, validation_split=0.05)
except KeyboardInterrupt as e:
    print("Interrupted, saving model!")
print("Saving model")
seq.save('conv2dlstm-test-trained.model')

print("Done")
# Testing the network on one movie
# feed it with the first 7 positions and then
# predict the new positions
which = 1004
track = noisy_movies[which][:7, ::, ::, ::]

print("Predicting new values")
for j in range(16):
    new_pos = seq.predict(track[np.newaxis, ::, ::, ::, ::])
    new = new_pos[::, -1, ::, ::, ::]
    track = np.concatenate((track, new), axis=0)
    print(f"Predicting {j} done...")

print("Done")

print("Creating images...")
# And then compare the predictions
# to the ground truth
track2 = noisy_movies[which][::, ::, ::, ::]
for i in range(15):
    fig = plt.figure(figsize=(10, 5))

    ax_model = fig.add_subplot(121)

    if i >= 7:
        ax_model.text(1, 3, 'Predictions', fontsize=15, color='w')
    else:
        ax_model.text(1, 3, 'Initial trajectory', fontsize=15, color='w')

    ax_model.imshow(track[i, ::, ::, 0])

    ax_gt = fig.add_subplot(122)
    ax_gt.text(1, 3, 'Ground truth', fontsize=15, color='w')

    toplot = track2[i, ::, ::, 0]
    if i >= 2:
        toplot = shifted_movies[which][i - 1, ::, ::, 0]

    plt.imshow(toplot)
    plt.savefig('%i_animate.png' % (i + 1))
    print(f"Saving image {i}")
