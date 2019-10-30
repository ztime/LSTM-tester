import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import Image

from pprint import pprint

HIGH_PIXEL_LEVEL = 100

def main():
    print("hej")
    parser = argparse.ArgumentParser()
    parser.add_argument('datafile', type=os.path.abspath, help='path to the datafile')
    args = parser.parse_args()

    frames = load_blob(args.datafile)
    pixels_per_frame = []
    for frame in frames:
        pixels_per_frame.append(count_pixels(frame)[0])
    print(f"Average white pixels per frame: {np.average(pixels_per_frame)}")
    high_frames = []
    for i in range(len(pixels_per_frame)):
        if pixels_per_frame[i] > HIGH_PIXEL_LEVEL:
            high_frames.append(i)
    print(f"Frames with more than {HIGH_PIXEL_LEVEL} white pixels: {high_frames}")
    dim_for_sub_plot = math.ceil(math.sqrt(len(high_frames)))
    high_frames_fig = plt.figure()
    sub_counter = 1
    for f in high_frames:
        formatted_frame = (frames[f] * 255).astype('uint8')
        img = Image.fromarray(formatted_frame)
        ax = high_frames_fig.add_subplot(dim_for_sub_plot, dim_for_sub_plot, sub_counter)
        ax.title.set_text(f"Frame {f}")
        ax.imshow(img)
        ax.axis('off')
        sub_counter += 1
    fig = plt.figure()
    plt.title("White pixel count per frame")
    plt.plot(range(len(pixels_per_frame)), pixels_per_frame)

    pixel_x_100, pixel_y_100 = pixel_wise_distances(frames[100])
    pixel_x_101, pixel_y_101 = pixel_wise_distances(frames[101])
    pixel_x_102, pixel_y_102 = pixel_wise_distances(frames[102])

    fig_pixel = plt.figure()
    plt.title("Histogram for frame 100-102")
    plt.plot(pixel_x_100, pixel_y_100, pixel_x_101, pixel_y_101, pixel_x_102, pixel_y_102)

    plt.show()

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
