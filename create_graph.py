import matplotlib.pyplot as plt
import argparse
from pprint import pprint

def main():
    parser = argparse.ArgumentParser(
            description="Create a graph from csv-style file.\nExpects first row to be headers, and each row after values"
            )
    parser.add_argument('csvfile', help="What file to use")
    parser.add_argument('-x', '--x_axis', help='What column to use as x axis, defaults to first column')
    parser.add_argument('-t', '--title', help='Title to display')
    args = parser.parse_args()

    legend_values_dict = parse_file(args.csvfile)

    if args.x_axis is None:
        # No axis choosen, take the first in the file
        args.x_axis = list(legend_values_dict)[0]
    # Find the x axis
    if args.x_axis not in legend_values_dict:
        print(f"Could not fix axis {args.x_axis} in file, aborting...")
        quit()
    x_axis = legend_values_dict.pop(args.x_axis)

    fig = plt.figure()
    if args.title:
        plt.title(args.title)
    for label, y_values in legend_values_dict.items():
        plt.plot(x_axis, y_values, label=label)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.subplots_adjust(right=0.75)
    plt.show()

def parse_file(csvfile: str) -> {}:
    with open(csvfile, 'r') as f:
        lines = f.readlines()
    if len(lines) < 2:
        print("Need more than two lines in file to create graph")
        quit()
    legend_dict = dict()
    # First line should contain name of the columns
    legends = lines[0].split()
    for legend in legends:
        legend_dict[legend] = []

    for i in range(1, len(lines)):
        values = lines[i].split()
        zipped = zip(legends, values)
        for (legend, val) in list(zipped):
            legend_dict[legend].append(val)

    return legend_dict



if __name__ == '__main__':
    main()

