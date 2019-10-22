from keras.layers import Input, LSTM, ConvLSTM2D, Conv3D, BatchNormalization
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard

from PIL import Image

import os
import sys
import numpy as np
import datetime
import subprocess
import random
from pprint import pprint

class Load_Noisy_LSTM():
    def __init__(
            self,
            path_to_model,
            folder_to_save_in,
            no_frames_to_generate,
            prefix,
            seed_data=None,
            seed_data_frame_offset=None,
            save_images=False,
            generate_video=False,
            save_zip=False,
            ):
        # Check arguments
        try:
            self.no_frames_to_generate = int(no_frames_to_generate)
        except ValueError as e:
            print(f"no frames to generate has to be an valid int (now:{no_frames_to_generate})")
            quit()
        print(f"Loading {path_to_model}...")
        self.model = self.load_keras_model(path_to_model)
        print("Done")
        self.check_folder(folder_to_save_in)
        self.prefix = prefix

        # Find out what the sequence length is in the model
        # There really must be a better way, but this is from get_config()
        #  {'layers': [{'class_name': 'ConvLSTM2D',
            # 'config': {'activation': 'tanh',
                # 'activity_regularizer': None,
                # 'batch_input_shape': (None, 25, 64, 64, 1),
                # }}]}                        ^-- there is the desired int!
        model_conf = self.model.get_config()
        layer_0_conf = model_conf.get('layers',[{}])[0].get('config')
        input_shape = layer_0_conf.get('batch_input_shape')
        self.seq_length = input_shape[1]
        self.img_width = input_shape[2]
        self.img_height = input_shape[3]
        self.img_channels = input_shape[4]

        # See if we want to generate random data for seed or not
        # TODO: For now just start with zeros
        if seed_data is None:
            # 1 is for number of sequences, since we are predicting it should only be one
            # self.seed_data = np.zeros((self.seq_length, self.img_width, self.img_height, self.img_channels))
            # self.seed_data = np.ones((self.seq_length, self.img_width, self.img_height, self.img_channels))
            self.seed_data = np.random.random((self.seq_length, self.img_width, self.img_height, self.img_channels))
        else:
            self.seed_data = self.load_seed_data(seed_data, seed_data_frame_offset, self.seq_length)
            # TODO: Load data here
            # self.seed_data = np.zeros((self.seq_length, self.img_width, self.img_height, self.img_channels))

        self.saved_frames = np.copy(self.seed_data)

        # Lets predict!
        # Seed data is the "stack" that we keep changing and passing to the predict
        # function, and saved_frames ...saves all the frames
        for i in range(self.no_frames_to_generate):
            if i % 10 == 0:
                print(f"Generating frame {i}...")
            # Add axis in front, as the predict function expects a sequence (like during training)
            new_frame = self.model.predict(self.seed_data[np.newaxis, ::, ::, ::, ::])
            # Remove the extra dimenson given by predict
            new = new_frame[::,-1,::,::,::]
            print(f"Max in frame {i}: {np.amax(new)}")
            # Remove first frame so our window always is seq_length
            self.seed_data = np.delete(self.seed_data, 0 , 0) # removing first frame
            # Add new frame, and save it 
            self.seed_data = np.concatenate((self.seed_data, new), axis=0)
            self.saved_frames = np.concatenate((self.saved_frames, new), axis=0)
        print(f"Done generating frames!")
        # Save it with numpy
        filename_npy_frames = os.path.join(self.folder_to_save_in, f"{self.prefix}-frames-compressed.npy")
        self.save_raw_frames_to_file(filename_npy_frames, self.saved_frames)

        # Images!
        if save_images:
            print("Generating images...")
            self.generate_images(self.saved_frames, self.folder_to_save_in, self.prefix)
            print("Done")
        # Video!
        if generate_video:
            print("Using ffmpeg to generate avi video...")
            commands = [
                    'ffmpeg',
                    '-y', # Overwrite files without asking
                    '-r', # Set framerate...
                    f"{self.seq_length}", # ...to seq_length
                    '-pattern_type', # Regextype ...
                    'glob', # ...set to global
                    f"-i", # Pattern to use when ...
                    f"'{self.folder_to_save_in}/*.png'", # ...looking for image files
                    f"{self.folder_to_save_in}/{self.prefix}-video.avi", # Where to save
                    ]
            print(f"Running command '{' '.join(commands)}'")
            subprocess.run(' '.join(commands), shell=True)
        # Zippity zappity
        if save_zip:
            print("Zipping new folder...")
            commands = [
                    'zip',
                    '-r', # Recursive
                    f"{self.folder_to_save_in}.zip", # Filename
                    f"{self.folder_to_save_in}" # Folder to zip
                    ]
            print(f"Running command '{' '.join(commands)}'")
            subprocess.run(' '.join(commands), shell=True)

    def generate_images(self, frames, folder_to_save_in, prefix):
        # fix for appending zeros for ffmpeg to filename
        no_frames = frames.shape[0]
        no_of_numbers = len(str(no_frames))
        for i in range(no_frames):
            # Add leading zeros 
            filename = f"{prefix}-frame-{i:0>{no_of_numbers}}.png"
            filename = os.path.join(folder_to_save_in, filename)
            single_frame = frames[i]
            single_frame_reshaped = np.reshape(single_frame, (single_frame.shape[0], single_frame.shape[1]))
            formatted_frame = (single_frame_reshaped * 255 / 1.0).astype('uint8')
            img = Image.fromarray(formatted_frame)
            img.save(filename, "PNG")

    def save_raw_frames_to_file(self, filename, frames):
        """
        Saves the frames to a file along with the shape
        using numpy.savez_compressed to save space (and packing the bits)
        since the space vs time to unpack is very favourable
        """
        # packed_frames = np.packbits(frames, axis=None)
        shape = frames.shape
        with open(filename, 'wb') as numpy_file:
            np.savez_compressed(numpy_file, frames=frames, shape=frames.shape)

    def check_folder(self, folder_to_save_in):
        if os.path.isdir(folder_to_save_in):
            y_n = input(f"Folder '{folder_to_save_in}' already exists, continue?[y/N]")
            if y_n != 'y':
                quit()
        else:
            os.mkdir(folder_to_save_in)
        self.folder_to_save_in = folder_to_save_in

    def load_keras_model(self, path_to_model):
        try:
            model = load_model(path_to_model)
        except ValueError as e:
            print(f"Could not load {path_to_model}: {e}")
        return model

    def load_seed_data(self, seed_data, seed_data_frame_offset, seq_length):
        loaded_frames = self.load_blob(seed_data)
        offset = 0
        try:
            offset = int(seed_data_frame_offset)
        except ValueError:
            # Offset could be None or random
            if seed_data_frame_offset is not None and seed_data_frame_offset.lower() == 'random':
                offset = random.randrange(loaded_frames.shape[0] - seq_length) # Shape[0] is number of frames
        selected_frames = loaded_frames[offset:offset+seq_length, ::, ::]
        # We need to add the last channel dimension (which is just 1 but tensorflow wants it)
        selected_frames = np.reshape(selected_frames, (seq_length, selected_frames.shape[1], selected_frames.shape[2], 1))
        return selected_frames

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

if __name__ == '__main__':
    seeds = [945, 100, 1500, 400]
    for seed in seeds:
        loaded_model = Load_Noisy_LSTM(
                    # "results_lstm_first_test/saved_models/model--32--1.00.hdf5", #path_to_model,
                    # "results_lstm_conv_rain_with_dropout/saved_models/model--10--1.00.hdf5", #path_to_model,
                    "../results_from_training/KL_test/saved_models/model--04--0.00.hdf5", #path_to_model,
                    f"../results_from_training/KL_test_model_04_real_seed_{seed}", #folder_to_save_in,
                    150, #no_frames_to_generate,
                    # 2, #no_frames_to_generate,
                    f"KL_test_model_04_real_seed_{seed}", #prefix,
                    seed_data="../numpy_dataset_128x128/bugs/20190423_1215_0_blobdata_part2.npy",
                    # seed_data_frame_offset=None,
                    # seed_data_frame_offset='random',
                    seed_data_frame_offset=seed,
                    # save_images=False,
                    save_images=True,
                    generate_video=True,
                    save_zip=True,
                    )
    # loaded_model = Load_Noisy_LSTM(
                # "results_lstm_bugs_chance/saved_models/model--26--1.00.hdf5", #path_to_model,
                # "predict_lstm_bugs_chance_random", #folder_to_save_in,
                # 300, #no_frames_to_generate,
                # # 2, #no_frames_to_generate,
                # "lstm_bugs_chance_random", #prefix,
                # seed_data=None,
                # seed_data_frame_offset=None,
                # # save_images=False,
                # save_images=True,
                # generate_video=False,
                # )
    # loaded_model = Load_Noisy_LSTM(
                # "results_lstm_conv_rain_with_dropout/saved_models/model--10--1.00.hdf5", #path_to_model,
                # "predict_lstm_conv_rain_w_dropout", #folder_to_save_in,
                # # 300, #no_frames_to_generate,
                # 2, #no_frames_to_generate,
                # "lstm_first_conv_rain_w_dropout", #prefix,
                # seed_data=None,
                # seed_data_frame_offset=None,
                # # save_images=False,
                # save_images=True,
                # generate_video=False,
                # )
