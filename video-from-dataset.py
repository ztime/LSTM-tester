from PIL import Image

import os
import sys
import numpy as np
import datetime
import subprocess
import random
import argparse
import tempfile
from pprint import pprint

def main():
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('datafile', type=os.path.abspath)
    parser.add_argument('-o', '--output', type=os.path.abspath)
    parser.add_argument('-f', '--framerate', type=int, default=25)

    args = parser.parse_args()

    # load all frames 
    all_frames = load_blob(args.datafile)
    print(f"Loaded all frames ({all_frames.shape[0]})")
    # Use filename as prefix for video
    _, filename = os.path.split(args.datafile)
    # Remove last .npy
    filename = filename.split('.')[0]
    # generate images in a temporary folder
    temporary_folder_obj = tempfile.TemporaryDirectory()
    temporary_folder = temporary_folder_obj.name
    print(f"Created temporary folder ({temporary_folder})")

    # Don't really need prefix, just do it anyway
    print(f"Started generating images...")
    generate_images(all_frames, temporary_folder, filename)
    print("Done")
    # generate video from that temporary folder
    print("Using ffmpeg to generate avi video...")
    commands = [
            'ffmpeg',
            '-y', # Overwrite files without asking
            '-r', # Set framerate...
            f"{args.framerate}", # ...to seq_length
            '-pattern_type', # Regextype ...
            'glob', # ...set to global
            f"-i", # Pattern to use when ...
            f"'{temporary_folder}/*.png'", # ...looking for image files
            ]
    # Append where to save
    if args.output is None:
        commands.append(f"{filename}.avi")
    else:
        commands.append(f"{args.output}/{filename}.avi")
    print(f"Running command '{' '.join(commands)}'")
    subprocess.run(' '.join(commands), shell=True)

    # clean up
    print("Cleaning up...")
    temporary_folder_obj.cleanup()
    print("Done!")

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

def generate_images(frames, folder_to_save_in, prefix):
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

if __name__ == '__main__':
    main()
