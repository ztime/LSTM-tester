from keras.datasets import mnist
from keras.layers import Input, Dense, Flatten, Reshape, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

from matplotlib import pyplot as plot

import numpy as np
import sys
import os
import datetime

class GAN():
    def __init__(
            self,
            data_path,
            data_no_files_to_load,
            name,
            folder_to_save_in
            ):
        if os.path.isdir(folder_to_save_in):
            y_n = input(f"Folder '{folder_to_save_in}' already exists, continue?[y/N]")
            if y_n != 'y':
                quit()
        else:
            os.mkdir(folder_to_save_in)
        self.folder_to_save_in = folder_to_save_in
        date_separator = datetime.datetime.now().strftime("%d%m%Y-%H%M%S")
        summary_file_name = f"{name}-summary-{date_separator}.log"
        self.summary_file = os.path.join(folder_to_save_in, summary_file_name)
        self.name = name
        self.write_to_summary(f"Started summary file '{summary_file_name}'...")
        self.x_train, self.y_train, self.x_test, self.y_test = self.load_data(data_path, data_no_files_to_load)
        # The training images / generated images
        self.img_height = self.x_train.shape[1]
        self.img_width = self.x_train.shape[2]
        self.img_channels = 1
        self.img_shape = (self.img_height, self.img_width, self.img_channels)
        # Input to generator
        self.latent_input = 256

        adam_learning_rate = 0.0002
        adam_clip = 0.5
        optimizer = Adam(adam_learning_rate, adam_clip)

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(
                loss='binary_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy']
                )
        self.generator = self.build_generator()
        z = Input(shape=(self.latent_input,))
        img = self.generator(z)

        self.discriminator.trainable = False

        validity = self.discriminator(img)

        self.combined = Model(z, validity)
        self.combined.compile(
                loss='binary_crossentropy',
                optimizer=optimizer
                )

    def write_to_summary(self, str_to_write):
        with open(self.summary_file, 'a') as f:
            f.write(f"{str_to_write}\n")
        print(str_to_write)

    def build_generator(self):
        """
        All layers/values from dcgan implementation
        """
        model = Sequential()

        # We have to scale correctly in CNN layers
        channels_to_up = 128
        res_number = self.img_height // 4
        magic_input_number = channels_to_up * res_number * res_number

        model.add(Dense(magic_input_number, activation="relu", input_dim=self.latent_input))
        model.add(Reshape((res_number,res_number,channels_to_up)))
        model.add(UpSampling2D())
        model.add(Conv2D(channels_to_up, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(channels_to_up // 2, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self.img_channels, kernel_size=3, padding="same"))

        model.summary(print_fn=self.write_to_summary)

        noise = Input(shape=(self.latent_input,))
        img = model(noise)

        return Model(noise, img)


    def build_discriminator(self):
        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.summary(print_fn=self.write_to_summary)
        img = Input(shape=self.img_shape)
        validity = model(img)
        return Model(img, validity)

    def load_data(self, data_path, max_files_to_load):
        """
        Convert the data into [-1, 1]
        """
        # 20 files = Approx 40 000 images
        no_files_loaded = 0
        # load everything in the folder
        x_train = False
        for root, dirs, files in os.walk(data_path):
            for one_file in files:
                self.write_to_summary(f"Loading from:{root}/{one_file}")
                file_path = os.path.join(root, one_file)
                if x_train is False:
                    x_train = self.load_blob(file_path)
                else:
                    more_data = self.load_blob(file_path)
                    x_train = np.concatenate((x_train, more_data), axis=0)
                    no_files_loaded += 1
                    if no_files_loaded >= max_files_to_load:
                        break
        # Throw the types around a bit
        x_train = x_train.astype(np.int8)
        # -1 is better (?) for gradients
        x_train[x_train == 0] = -1
        x_train = x_train.astype(np.single)
        # No validation for now
        x_test = None
        y_train = None
        y_test = None
        self.write_to_summary(f"Loaded {x_train.shape[0]} images into x_train!")
        return (x_train, y_train, x_test, y_test)

    def load_blob(self, abspath):
        """
        Loads a blob file and converts it
        """
        loaded_numpy = np.load(abspath)
        loaded_frames = loaded_numpy['frames']
        loaded_shape = loaded_numpy['shape']
        no_frames = loaded_shape[0]
        width = loaded_shape[1]
        height = loaded_shape[2]
        frames_unpacked = np.unpackbits(loaded_frames)
        frames_unpacked = frames_unpacked.reshape((no_frames, width, height))
        return frames_unpacked

    def train(self, epochs, batch_size=128, sample_interval=50):
        X_train = np.expand_dims(self.x_train, axis=3)

        # ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        # Keep losses and stuff
        old_losses = []
        old_losses.append(("Disc-loss", "Disc-acc", "Gen-loss"))

        for epoch in range(epochs):

            # Train discriminator with fakes and valids
            random_ids = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[random_ids]

            noise = np.random.normal(0, 1, (batch_size, self.latent_input))
            # Turn this into binary vector? Since thats what the input data looks like...
            # But might just reduce complexity of the generated images..
            noise = np.zeros((batch_size, self.latent_input))

            # Generate fakes
            gen_imgs = self.generator.predict(noise)

            # trainit!
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Train generator
            noise = np.random.normal(0, 1, (batch_size, self.latent_input))
            g_loss = self.combined.train_on_batch(noise, valid)

            print(f"Epoch:{epoch} [D loss:{d_loss[0]:.4f}, acc:{100*d_loss[1]:.4f}] [G loss:{g_loss:.4f}]")
            old_losses.append((d_loss[0], d_loss[1], g_loss))
            if epoch % sample_interval == 0:
                self.sample_images(epoch)
        # Done
        with open(os.path.join(self.folder_to_save_in, f'loss-{self.name}.log'), 'a') as f:
            first = True
            for l in old_losses:
                if first:
                    f.write(f"{l[0]}\t{l[1]}\t{l[2]}\n")
                    first = False
                else:
                    f.write(f"{l[0]:.6f}\t{100*l[1]:.6f}\t{l[2]:.6f}\n")

    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_input))
        gen_imgs = self.generator.predict(noise)
        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plot.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        folder_path = os.path.join(self.folder_to_save_in, 'images')
        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)
        fig.savefig(f"{folder_path}/{self.name}-epoch-{epoch}.png")
        plot.close()

if __name__ == '__main__':
    gan = GAN(
            "/home/exjobb/style_transfer/numpy_dataset_64x64/water",
            20,
            "testing_water",
            "results_with_water")
    gan.train(3000, batch_size=32)
