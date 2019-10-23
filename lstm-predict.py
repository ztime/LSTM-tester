from keras.models import load_model

from PIL import Image

import os
import sys
import numpy as np
import datetime
import subprocess
import random
import argparse
import tempfile

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', type=os.path.abspath, help='Folder to store results in', required=True)
    parser.add_argument('-p', '--prefix', help='Prefix for the files', required=True)
    parser.add_argument('-m', '--model', help='What model to generate from (hdf5 file)', required=True)
    parser.add_argument('-s', '--seed_data', help='What seed data to use', choices=['zeros', 'ones', 'random', 'real'], required=True)
    parser.add_argument('-f', '--frames', type=int, help='How many frames to generate (full length will be sequence_length + frames)', required=True)
    parser.add_argument('--real_seed_path', type=os.path.abspath, help="Path to seed data to use, required if seed_data = real")
    parser.add_argument('--real_seed_offset', type=int, default=0, help="Offset in the real seed file to start sampling from")
    parser.add_argument('--save_numpy_file', action='store_true', default=False, help='Store all frames as numpy file')
    parser.add_argument('--save_images', action='store_true', default=False, help='Store all images in result folder')
    parser.add_argument('--save_video', action='store_true', default=False, help='Store an video in results folder')
    parser.add_argument('--save_zip', action='store_true', default=False, help='Zip the result-folder for easy transfer')
    parser.add_argument('--framerate', type=int, default=25, help="Framerate for generating video, not to be confused with frames")

    args = parser.parse_args()

    # Manually parse some arguments
    if args.seed_data == 'real' and args.real_seed_path is None:
        parser.error('--real_seed_path is required when using -s/--seed_data.')

    if not args.save_numpy_file and not args.save_images and not args.save_video:
        parser.error('one of --save_numpy_file, --save_images, --save_video must be active')

    cprint("Loading model...")
    model = load_keras_model(args.model)
    cprint("Loaded model!", print_green=True)
    check_folder(args.output)
    seq_length, img_width, img_height, img_channels = get_dimensions_from_model(model)

    seed_data_shape = (seq_length, img_width, img_height, img_channels)
    seed_data = None
    # No need for else, determined by argument parser
    if args.seed_data == 'zeros':
        seed_data = np.zeros(seed_data_shape)
        cprint("Seed data is filled with zeros")
    elif args.seed_data == 'ones':
        seed_data = np.ones(seed_data_shape)
        cprint("Seed data is filled with ones")
    elif args.seed_data == 'random':
        seed_data = np.random.random(seed_data_shape)
        cprint("Seed data is random")
    elif args.seed_data == 'real':
        seed_data = load_real_seed_data(args.real_seed_path, args.real_seed_offset, seq_length)
        cprint("Seed data is loaded from real data")
    cprint("Generating frames...")

    # These are all the frames loaded AND generated, seed_data is used as a stack
    # to predict new data
    saved_frames = np.copy(seed_data)
    # Lets predict!
    # Seed data is the "stack" that we keep changing and passing to the predict
    # function, and saved_frames ...saves all the frames
    for i in range(args.frames):
        if i % 10 == 0:
            cprint(f"Generating frame {i}...")
        # Add axis in front, as the predict function expects a sequence (like during training)
        new_frame = model.predict(seed_data[np.newaxis, ::, ::, ::, ::])
        # Remove the extra dimenson given by predict
        new = new_frame[::,-1,::,::,::]
        max_in_frame = np.amax(new)
        # 0.02 is just a number choosen because it showed up in training
        # often when max is less than that, the frame is empty
        if max_in_frame < 0.02:
            cprint(f"Warning: Max in frame {i}: {max_in_frame}", print_red=True)
        # Remove first frame so our window always is seq_length
        seed_data = np.delete(seed_data, 0 , 0) # removing first frame
        # Add new frame, and save it 
        seed_data = np.concatenate((seed_data, new), axis=0)
        saved_frames = np.concatenate((saved_frames, new), axis=0)
    cprint(f"Done generating frames!", print_green=True)

    if args.save_numpy_file:
        cprint("Saving numpy file...")
        filename_npy = os.path.join(args.output, f"{args.prefix}-frames-compressed.npy")
        save_raw_frames_to_file(filename_npy, saved_frames)
        cprint("Saved numpy file!", print_green=True)

    # Next part is a bit special, if we only want the generated video
    # we still need the image frames. 
    # if we dont want any of it, just quit
    if not args.save_images and not args.save_video:
        cprint("--save_images and --save_video was false, nothing more to do, quitting!", print_green=True)
        quit()
    # So:
    # if we want both images and video: Save images to regular folder
    # only video: Save images to temporary folder
    # Defaults
    save_images_temporary = False
    save_images_folder = args.output
    if args.save_images is False and args.save_video is True:
        save_images_temporary = True
        save_images_folder_obj = tempfile.TemporaryDirectory()
        save_images_folder = save_images_folder_obj.name

    cprint("Generating images to {save_images_folder}...")
    save_images_to_folder(saved_frames, save_images_folder, args.prefix)
    cprint("Generated images!", print_green=True)

    if args.save_video:
        cprint("Using ffmpeg to generate avi video...")
        commands = [
                'ffmpeg',
                '-y', # Overwrite files without asking
                '-r', # Set framerate...
                f"{args.framerate}", # ...to seq_length
                '-pattern_type', # Regextype ...
                'glob', # ...set to global
                f"-i", # Pattern to use when ...
                f"'{save_images_folder}/*.png'", # ...looking for image files
                f"{args.output}/{args.output}-video.avi", # Where to save
                ]
        cprint(f"Running command '{' '.join(commands)}'")
        subprocess.run(' '.join(commands), shell=True)
        cprint("Dont generating video!", print_green=True)
    # Clean up if we were using temporary folder for the images
    if save_images_temporary:
        cprint(f"Cleaning up temporary folder {save_images_folder}...")
        save_images_folder_obj.cleanup()
        cprint("Cleanup done.", print_green=True)
    # zippety zappety
    if args.save_zip:
        cprint(f"Zipping folder '{args.output}'...")
        commands = [
                'zip',
                '-r', # Recursive
                f"{args.output}/{args.output}.zip", # Filename
                f"{args.output}" # Folder to zip
                ]
        cprint(f"Running command '{' '.join(commands)}'")
        subprocess.run(' '.join(commands), shell=True)
    cprint(f"Finished, output saved in '{args.output}'!", print_green=True)

def save_images_to_folder(frames, folder_to_save_in, prefix):
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

def save_raw_frames_to_file(filename, frames):
    """
    Saves the frames to a file along with the shape
    using numpy.savez_compressed to save space
    since the space vs time to unpack is very favourable
    """
    # Dont pack the bits, we are dealing with floats
    # packed_frames = np.packbits(frames, axis=None)
    shape = frames.shape
    with open(filename, 'wb') as numpy_file:
        np.savez_compressed(numpy_file, frames=frames, shape=frames.shape)

def load_real_seed_data(path_to_seed_data, offset, seq_length):
    loaded_frames = load_blob(path_to_seed_data)
    selected_frames = loaded_frames[offset:offset+seq_length, ::, ::]
    # We need to add the last channel dimension (which is just 1 but tensorflow wants it)
    selected_frames = np.reshape(selected_frames, (seq_length, selected_frames.shape[1], selected_frames.shape[2], 1))
    return selected_frames

def load_blob(abspath):
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

def get_dimensions_from_model(model):
    # There really must be a better way, but this is uses get_config()
    #  {'layers': [{'class_name': 'ConvLSTM2D',
        # 'config': {'activation': 'tanh',
            # 'activity_regularizer': None,
            # 'batch_input_shape': (None, 25, 64, 64, 1),
            # }}]}                        ^-- there is the desired int!
    model_conf = model.get_config()
    layer_0_conf = model_conf.get('layers',[{}])[0].get('config')
    input_shape = layer_0_conf.get('batch_input_shape')
    seq_length = input_shape[1]
    img_width = input_shape[2]
    img_height = input_shape[3]
    img_channels = input_shape[4]
    return seq_length, img_width, img_height, img_channels

def check_folder(folder_to_save_in):
    if os.path.isdir(folder_to_save_in):
        cprint(f"Folder '{folder_to_save_in}' already exists, continue?[y/N]")
        y_n = input()
        if y_n != 'y':
            quit()
    else:
        os.mkdir(folder_to_save_in)

def load_keras_model(path_to_model):
    try:
        model = load_model(path_to_model)
    except (ValueError, OSError) as e:
        cprint(f"Could not load {path_to_model}: {e}", print_red=True)
        quit()
    return model

def cprint(string, print_red=False, print_green=False):
    """
    Used because Tensorflow prints so much garbage, it's hard to see
    what is printed by the script and not.
    Defaults to printing warning (yellow-ish) color.
    """
    if print_red:
        print(f'\033[91m{string}\033[0m')
    elif print_green:
        print(f'\033[92m{string}\033[0m')
    else:
        print(f'\033[93m{string}\033[0m')
