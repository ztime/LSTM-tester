import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import Image

from pprint import pprint

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('datafile', type=os.path.abspath, help='path to the datafile')
    parser.add_argument('--use_mnist', action='store_true')
    parser.add_argument('--frames_to_use', type=int, default=1000)
    parser.add_argument('--view_high_frames', action='store_true')
    parser.add_argument('--high_frame_level', type=int, default=10000)
    parser.add_argument('--show_histograms', action='store_true')
    parser.add_argument('--histogram_frames', default='0-2')
    parser.add_argument('--show_frames', action='store_true')
    parser.add_argument('--show_frames_index', default='0-10')
    parser.add_argument('--show_frames_grid')
    args = parser.parse_args()

    if args.use_mnist:
        print("Loading MNIST data...")
        frames = load_data_moving_mnist(args.frames_to_use)
    else:
        print(f"Loading blob {args.datafile}...")
        frames = load_blob(args.datafile)
    print("Done.")
    print(f"Shape of data: {frames.shape}")

    pixels_per_frame = []
    for frame in frames:
        # [0] is to the the number of white pixels
        pixels_per_frame.append(count_pixels(frame)[0])
    print(f"Average white pixels per frame: {np.average(pixels_per_frame)}")
    high_frames = []
    for i in range(len(pixels_per_frame)):
        if pixels_per_frame[i] > args.high_frame_level:
            high_frames.append(i)
    print(f"Frames with more than {args.high_frame_level} white pixels: {high_frames}")
    if args.view_high_frames:
        print("Viewing high frames...")
        dim_for_sub_plot = math.ceil(math.sqrt(len(high_frames)))
        high_frames_fig = plt.figure(1)
        sub_counter = 1
        for f in high_frames:
            formatted_frame = (frames[f] * 255).astype('uint8')
            img = Image.fromarray(formatted_frame)
            ax = high_frames_fig.add_subplot(dim_for_sub_plot, dim_for_sub_plot, sub_counter)
            ax.title.set_text(f"Frame {f}")
            ax.imshow(img)
            ax.axis('off')
            sub_counter += 1
        plt.title("White pixel count per frame")
        plt.plot(range(len(pixels_per_frame)), pixels_per_frame)
        print("Done")

    if args.show_histograms:
        print("Viewing histograms...")
        # Parse argument to know what frames
        start_frame = args.histogram_frames.split('-')[0]
        end_frame = args.histogram_frames.split('-')[-1]
        start_frame = int(start_frame)
        end_frame = int(end_frame)
        fig_pixel = plt.figure(2)
        plt.title(f"Histogram for frame {args.histogram_frames}")
        for frame in range(start_frame, end_frame + 1):
            p_x, p_y = pixel_wise_distances(frames[frame])
            plt.plot(p_x, p_y)
        print("Done")

    if args.show_frames:
        # Parse argument to know what frames
        print("Viewing frames...")
        start_frame = args.show_frames_index.split('-')[0]
        end_frame = args.show_frames_index.split('-')[-1]
        start_frame = int(start_frame)
        end_frame = int(end_frame)
        frames_as_list = [ i for i in range(start_frame, end_frame + 1) ]
        # if we have a supplied grid size, use that, otherwise
        # we create a square thats big enough to fit them all
        dim_for_sub_1 = None
        dim_for_sub_2 = None
        if args.show_frames_grid is None:
            dim_for_sub= math.ceil(math.sqrt(len(frames_as_list)))
            dim_for_sub_1 = dim_for_sub
            dim_for_sub_2 = dim_for_sub
        else:
            dim_for_sub_1 = int(args.show_frames_grid.split('x')[0])
            dim_for_sub_2 = int(args.show_frames_grid.split('x')[-1])
        # Lets do this!
        frames_fig = plt.figure(3)
        frames_fig.tight_layout()
        sub_counter = 1
        for f in frames_as_list:
            formatted_frame = (frames[f] * 255).astype('uint8')
            img = Image.fromarray(formatted_frame)
            ax = frames_fig.add_subplot(dim_for_sub_1, dim_for_sub_2, sub_counter)
            ax.title.set_fontsize(8)
            # ax.title.set_text(f"Frame {f}")
            ax.title.set_text(f"Frame {sub_counter}")
            ax.imshow(img, cmap='gray')
            ax.axis('off')
            sub_counter += 1
        # plt.title(f"Display of frames {args.show_frames_index}")
        # plt.plot(range(len(pixels_per_frame)), pixels_per_frame)
        print("Done")

    plt.show()

def load_data_moving_mnist(frames_to_load):
    """
    Load the moving mnist dataset used in the lstm paper

    Assumes that the file mnist_test_seq.npy is in the same folder
    as this script

    Can be downloaded from:
        http://www.cs.toronto.edu/~nitish/unsupervised_video/
    Maximum sequence length is 20 (so max use is 19) and available sequences are 10 000
    """
    MNIST_DATA_PATH = 'mnist_test_seq.npy'
    loaded_numpy = np.load(MNIST_DATA_PATH)
    _,total_loaded_sequences, width, height = loaded_numpy.shape
    frames = np.zeros((frames_to_load, width, height))
    # Wierd-ass format (frame, sequence, width, height) !?
    # Reformat to sane human values (sequence, frame, width, height)
    # and shape the sequence length at the same time 
    # also add an extra dimension 
    frame_in_seq_count = 0
    seq_count = 0
    for frame in range(frames_to_load):
        frames[frame] = loaded_numpy[frame_in_seq_count, seq_count, ::, ::]
        frame_in_seq_count += 1
        if frame_in_seq_count >= 20:
            seq_count += 1
        frame_in_seq_count = frame_in_seq_count % 20 #seq length is 20

    # Adjust the values
    frames /= 255.0
    frames[frames >= .5] = 1.
    frames[frames < .5] = 0.

    return frames

def pixel_wise_distances(frame):
    """
    Counts the distance between all pixels in frame
    """
    pixel_locations = []
    rows, cols = frame.shape
    for i in range(rows):
        for j in range(cols):
            if frame[i][j] == 1:
                pixel_locations.append((i,j))
    # Maximized range from 0,0 -> rows, cols
    max_length = math.ceil(math.sqrt((rows*rows) + (cols*cols)))
    distances = {}
    for pixel_index in range(len(pixel_locations)):
        x_1, x_2 = pixel_locations[pixel_index]
        for other_pixel in range(pixel_index + 1, len(pixel_locations)):
            y_1, y_2 = pixel_locations[other_pixel]
            dist = euclidian(x_1, x_2, y_1, y_2)
            if dist not in distances:
                distances[dist] = 0
            distances[dist] += 1
    # lists for the plot
    x = []
    y = []
    # No zero indexed array, so we use + 1
    for i in range(1, max_length + 1):
        x.append(i)
        if i not in distances:
            y.append(0)
        else:
            y.append(distances[i])

    return x, y

def euclidian(x_1, x_2, y_1, y_2):
    e = math.sqrt( (x_1 - y_1)**2 + (x_2 - y_2)**2 )
    return math.ceil(e)

def count_pixels(frame):
    """
    Returns a pixel count like this
    (white pixels, black pixels)

    Data is binary so ones and zeros
    """
    rows, cols = frame.shape
    white_pixels = frame.sum()
    return white_pixels, int((rows * cols) - white_pixels)

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

if __name__ == '__main__':
    main()
