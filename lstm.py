from keras.layers import Input, LSTM, ConvLSTM2D, Conv3D, BatchNormalization, Dense, Flatten
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard

import os
import sys
import numpy as np
import datetime

class Noisy_LSTM():
    def __init__(
            self,
            data_path,
            sequence_length,
            no_sequences,
            hidden_units,
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
            self.x_train, self.y_train = self.load_data(data_path, sequence_length, no_sequences)
            if self.x_train is False or self.y_train is False:
                self.write_to_summary(f"Couldn't load any data from {data_path}, aborting!")
                quit(1)
            # The training images 
            self.img_height = self.x_train[0].shape[1]
            self.img_width = self.x_train[0].shape[2]
            self.img_channels = 1
            self.img_shape = (self.img_height, self.img_width, self.img_channels)
            self.hidden_units = hidden_units
            self.sequence_length = sequence_length

            self.network = self.build_network()

            self.network.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
            self.network.summary()
            # setup callbacks
            model_folder = os.path.join(self.folder_to_save_in, 'saved_models')
            model_filename = os.path.join(self.folder_to_save_in, 'saved_models', 'model--{epoch:02d}--{val_acc:.2f}.hdf5')
            if not os.path.isdir(model_folder):
                os.mkdir(model_folder)
            self.model_callback = ModelCheckpoint(model_filename, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
            # self.model_callback = ModelCheckpoint(model_filename, verbose=1, save_best_only=True)
            tensorboard_filepath = os.path.join(self.folder_to_save_in, 'tensorboard_logs')
            # self.tensorboard_callback = TensorBoard(log_dir=tensorboard_filepath, histogram_freq=5, write_grads=True, write_graph=True, write_images=True)
            self.tensorboard_callback = TensorBoard(log_dir=tensorboard_filepath, histogram_freq=0, write_graph=True, write_images=True)

    def train(self):
        self.network.fit(
                self.x_train,
                self.y_train,
                batch_size=10,
                epochs=150,
                validation_split=0.1,
                callbacks=[self.model_callback, self.tensorboard_callback],
                )

    def build_network(self):
        model = Sequential()
        model.add(
                Flatten(
                    input_shape=(self.sequence_length, self.img_width, self.img_height, 1),
                    )
                )
        model.add(
                LSTM(
                    1024,
                    return_sequences=True,
                    )
            )
        model.add(BatchNormalization())
        model.add(
                LSTM(
                    1024,
                    return_sequences=True,
                    )
            )
        model.add(BatchNormalization())
        model.add(Reshape((self.sequence_length, self.img_width, self.img_height, 1)))
        return model

    def load_data(self, data_path, sequence_length, no_sequences=None):
        """
        Convert the data into training data for lstm
        total amount loaded will be sequence_size * no_sequences
        no_sequences will default to *load everything*

        No order can be guaranteed because of the os.walk not guaranteeing order

        sequence length of 4 will generate
        x = [[1,2,3]] y = [4]
        i.e sequence_lengths are inclusive
        """
        frames_available = 0
        # load everything in the folder
        all_data = False
        for root, dirs, files in os.walk(data_path):
            for one_file in files:
                self.write_to_summary(f"Loading from:{root}/{one_file}")
                file_path = os.path.join(root, one_file)
                if all_data is False:
                    all_data = self.load_blob(file_path)
                    frames_available += all_data.shape[0]
                    if frames_available // sequence_length > no_sequences:
                        break
                else:
                    more_data = self.load_blob(file_path)
                    all_data = np.concatenate((all_data, more_data), axis=0)
                    frames_available += more_data.shape[0]
                    if frames_available // sequence_length > no_sequences:
                        break
        if all_data is False:
            return (False,False)
        # Check how many sequences we will get
        final_no_sequences = frames_available // sequence_length
        if final_no_sequences > no_sequences:
            final_no_sequences = no_sequences
        else:
            final_no_sequences -= 1
        img_width = all_data.shape[1]
        img_height = all_data.shape[2]
        # Load frames into sequences and ground truths
        current_frame = 0
        current_sequence = 0
        # -1 in sequence_length becasue the final frame is in the ground truth
        x_train = np.zeros((final_no_sequences, sequence_length, img_width, img_height, 1))
        y_train = np.zeros((final_no_sequences, sequence_length, img_width, img_height, 1))
        while True:
            training_frames = all_data[current_frame: current_frame + sequence_length]
            truth_frame = all_data[current_frame + 1: current_frame + sequence_length + 1]
            current_frame += sequence_length
            x_train[current_sequence] = np.expand_dims(training_frames, axis=3)
            y_train[current_sequence] = np.expand_dims(truth_frame, axis=3)
            current_sequence += 1
            if no_sequences is not None and current_sequence >= final_no_sequences:
                break
        # No validation for now
        self.write_to_summary(f"Loaded {len(x_train)} sequences of length {sequence_length}!")
        return (x_train, y_train)

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

    def write_to_summary(self, str_to_write):
        with open(self.summary_file, 'a') as f:
            f.write(f"{str_to_write}\n")
        print(str_to_write)

if __name__ == '__main__':
    lstm = Noisy_LSTM(
             "/home/exjobb/style_transfer/numpy_dataset_64x64/rain",
            25,
            2000,
            50,
            "lstm_non_conv_rain",
            "results_lstm_non_conv_rain")
    # lstm.train(3000, batch_size=32)
    lstm.train()
