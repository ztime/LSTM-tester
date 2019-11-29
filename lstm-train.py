import argparse
import os
import importlib
import numpy as np
import datetime

from keras.callbacks import ModelCheckpoint, TensorBoard, Callback
from keras.utils import plot_model

from pprint import pprint

# Globals... :'(
SUMMARY_FILE = None

# Discard frames that are > DISCARD_CONSTANT * average in pixel density
DISCARD_CONSTANT = 4

def main():
    global SUMMARY_FILE

    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', type=os.path.abspath, help='Where to store logs/models', required=True)
    parser.add_argument('-p', '--prefix', help='Prefix for naming all files', required=True)
    parser.add_argument('-m', '--modelfile', type=os.path.abspath, help='What model to use', required=True)
    parser.add_argument('-e', '--epochs', type=int, help='Number of epochs to train', required=True)
    parser.add_argument('-d', '--datapath', type=os.path.abspath, help='Path to training data', required=True)
    parser.add_argument('-b', '--batchsize', type=int, help="What batchsize to use", required=True)
    parser.add_argument('--sequence_length', type=int, default=25, help='How long sequence length in lstm')
    parser.add_argument('--no_of_sequences', type=int, default=-1, help='How many sequences to load, default is all')
    parser.add_argument('--save_tensorboard', action='store_true', help='Save tensorboard logs')
    parser.add_argument('--validation_split', type=float, default=0.2, help="Split training data into validation ratio")
    parser.add_argument('--save_metrics_each_batch', help='Save metrics after each batch to file')
    parser.add_argument('--save_plot_model', action='store_true', help='Save an image plot of the model')
    parser.add_argument('--yes_to_all', action='store_true', help='Answer yes to all questions!')
    parser.add_argument('--use_moving_mnist', action='store_true', help='Use the moving mnist dataset (overrides)')

    args = parser.parse_args()

    # Set up summary file
    date_separator = datetime.datetime.now().strftime("%d%m%Y-%H%M%S")
    summary_file_name = f"{args.prefix}-summary-{date_separator}.log"
    SUMMARY_FILE = os.path.join(args.output, summary_file_name)

    # Check folder
    if os.path.isdir(args.output):
        if args.yes_to_all:
            cprint(f"{args.output} already exists, using it anyway (thx to --yes_to_all flag)")
            y_n = 'y'
        else:
            y_n = input(f"'{args.output}' already exists, continue anyway?[y/N]")
        if y_n != 'y':
            quit()
    else:
        os.mkdir(args.output)

    # Before anything, we need to check if the model has something to override
    write_to_summary("Loading model...")
    # load model - go from path to file to module -> filename - relative path
    working_dir = os.getcwd()
    # Remove current directory
    model_import = args.modelfile[len(working_dir) + 1:].split('/')
    # Remove ".py" , yupp it's horrible
    model_import[-1] = os.path.splitext(model_import[-1])[0]
    # go to model.filename
    model_import = '.'.join(model_import)
    loaded_model = importlib.import_module(model_import)
    # We can let the model override some parameters
    model_has_data_prepare = False
    if hasattr(loaded_model, 'MODEL_OVERRIDES'):
        for param_name, param_value in loaded_model.MODEL_OVERRIDES.items():
            if param_name == 'sequence_length':
                cprint(f"FYI: Model override the sequence length from {args.sequence_length} to {param_value}", print_red=True)
                args.sequence_length = param_value
            elif param_name == 'batchsize':
                cprint(f"FYI: Model override batchsize from {args.batchsize} to {param_value}", print_red=True)
                args.batchsize = param_value
            elif param_name == 'data_prepare':
                cprint(f"FYI: Model has a function to override data reshape before training", print_red=True)
                model_has_data_prepare = True
            else:
                # Fallback
                cprint(f"Warning, model tried to override {param_name} but that is not supported!", print_red=True)


    # Check datapath, load data bc we need to know image dimensions of the trainingdata
    if args.no_of_sequences == -1:
        args.no_of_sequences = None
    if args.use_moving_mnist:
        x_train, y_train = load_data_moving_mnist(args.sequence_length, args.no_of_sequences)
    else:
        x_train, y_train = load_data(args.datapath, args.sequence_length, args.no_of_sequences)
    if x_train is False or y_train is False:
        cprint("Could not load data! Aborting...", print_red=True)
        quit()
    img_width, img_height = x_train[0].shape[1], x_train[0].shape[2]

    write_to_summary("Loaded data, folder created.")

    model = loaded_model.get_model(args.sequence_length, img_width, img_height)
    desc = loaded_model.get_description()
    write_to_summary("Loaded model successfully!", print_green=True)
    write_to_summary("Summary:")
    model.summary(print_fn=lambda x: write_to_summary(x))
    write_to_summary("Description:")
    write_to_summary(desc)

    if args.save_plot_model:
        plot_file = os.path.join(args.output, f"{args.prefix}-plot-model.png")
        plot_model(model, show_shapes=True, to_file=plot_file)
        cprint(f"Saved model image to {plot_file}!", print_green=True)
        quit()
    # Setup callbacks
    # Model saver
    model_folder = os.path.join(args.output, 'saved_models')
    model_filename = os.path.join(args.output, 'saved_models', 'model--{epoch:02d}--{val_loss:.2f}.hdf5')
    if not os.path.isdir(model_folder):
        os.mkdir(model_folder)
    model_callback = ModelCheckpoint(model_filename, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    callbacks = [model_callback]
    # Tensorboard
    if args.save_tensorboard:
        tensorboard_filepath = os.path.join(args.output, 'tensorboard_logs')
        tensorboard_callback = TensorBoard(
                log_dir=tensorboard_filepath,
                histogram_freq=2,
                batch_size=1,
                write_grads=True,
                write_graph=True,
                write_images=True,
                update_freq='epoch',
                )
        callbacks.append(tensorboard_callback)
    # Save metrics after each batch to file
    if args.save_metrics_each_batch is not None:
        from pprint import pprint
        import copy

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

        batch_file = os.path.join(args.output, args.save_metrics_each_batch)
        batch_metrics = BatchMetrics(batch_file)
        callbacks.append(batch_metrics)

    # Training time
    if model_has_data_prepare:
        # To allow models with several output or wierd
        # data transformations
        x_train, y_train = loaded_model.data_prepare(x_train, y_train)
        write_to_summary("Reshaped/reconfigured training data bc model had overwritten it")
    try:
        history = model.fit(
                x_train,
                y_train,
                batch_size=args.batchsize,
                epochs=args.epochs,
                validation_split=args.validation_split,
                shuffle=True,
                callbacks=callbacks,
                )
    except KeyboardInterrupt:
        # Save the model as is, with special filename
        write_to_summary("Training interrupted, attempting to save model...", print_red=False)
        model_filename = os.path.join(args.output, 'saved_models', 'model--Interrupted--saved.hdf5')
        model.save(model_filename)
        write_to_summary(f"Saved model to {model_filename}", print_green=True)
        quit()
    # Save the last model
    model_filename = os.path.join(args.output, 'saved_models', f'model--last--{args.epochs}--.hdf5')
    write_to_summary(f"Training complete, saving last model as {model_filename}", print_green=True)
    model.save(model_filename)


def load_data_moving_mnist(sequence_length, no_sequences=None):
    """
    Load the moving mnist dataset used in the lstm paper

    Assumes that the file mnist_test_seq.npy is in the same folder
    as this script

    Can be downloaded from:
        http://www.cs.toronto.edu/~nitish/unsupervised_video/
    Maximum sequence length is 20 (so max use is 19) and available sequences are 10 000
    """
    MNIST_DATA_PATH = 'mnist_test_seq.npy'
    assert(sequence_length <= 19)
    assert(no_sequences is None or no_sequences <= 10000)
    write_to_summary("Loading moving mnist dataset...")
    loaded_numpy = np.load(MNIST_DATA_PATH)
    _,total_loaded_sequences, width, height = loaded_numpy.shape
    write_to_summary(f"Loaded shape:{loaded_numpy.shape}")
    if no_sequences is None:
        no_sequences = total_loaded_sequences
    x_train = np.zeros((no_sequences, sequence_length, width, height, 1))
    y_train = np.zeros((no_sequences, 1, width, height, 1))
    # Wierd-ass format (frame, sequence, width, height) !?
    # Reformat to sane human values (sequence, frame, width, height)
    # and shape the sequence length at the same time 
    # also add an extra dimension 
    for seq_count in range(no_sequences):
        for frame_count in range(sequence_length):
            x_train[seq_count, frame_count] = loaded_numpy[frame_count, seq_count, ::, ::, np.newaxis]
        y_train[seq_count, 0] = loaded_numpy[frame_count + 1, seq_count, ::, ::, np.newaxis]

    # Adjust the values
    x_train /= 255.0
    y_train /= 255.0
    x_train[x_train >= .5] = 1.
    x_train[x_train < .5] = 0.
    y_train[y_train >= .5] = 1.
    y_train[y_train < .5] = 0.

    return x_train, y_train
    # Save for later
    """
    import matplotlib.pyplot as plt
    fig = plt.figure(1)
    ax = fig.add_subplot(4,2,1)
    ax.imshow(loaded_numpy[0,0], cmap='gray')
    ax = fig.add_subplot(4,2,3)
    ax.imshow(loaded_numpy[1,0], cmap='gray')
    ax = fig.add_subplot(4,2,5)
    ax.imshow(loaded_numpy[2,0], cmap='gray')
    ax = fig.add_subplot(4,2,7)
    ax.imshow(loaded_numpy[3,0], cmap='gray')

    ax = fig.add_subplot(4,2,2)
    ax.imshow(loaded_numpy[0,1], cmap='gray')
    ax = fig.add_subplot(4,2,4)
    ax.imshow(loaded_numpy[1,1], cmap='gray')
    ax = fig.add_subplot(4,2,6)
    ax.imshow(loaded_numpy[2,1], cmap='gray')
    ax = fig.add_subplot(4,2,8)
    ax.imshow(loaded_numpy[3,1], cmap='gray')

    plt.tight_layout()
    plt.show()

    fig = plt.figure(2)
    ax = fig.add_subplot(4,2,1)
    ax.imshow(x_train[0,7][:, :, 0], cmap='gray')
    ax = fig.add_subplot(4,2,3)
    ax.imshow(x_train[0,8][:, :, 0], cmap='gray')
    ax = fig.add_subplot(4,2,5)
    ax.imshow(x_train[0,9][:, :, 0], cmap='gray')
    ax = fig.add_subplot(4,2,7)
    ax.imshow(y_train[0,0][:, :, 0], cmap='gray')

    ax = fig.add_subplot(4,2,2)
    ax.imshow(x_train[8,7][:, :, 0], cmap='gray')
    ax = fig.add_subplot(4,2,4)
    ax.imshow(x_train[8,8][:, :, 0], cmap='gray')
    ax = fig.add_subplot(4,2,6)
    ax.imshow(x_train[8,9][:, :, 0], cmap='gray')
    ax = fig.add_subplot(4,2,8)
    ax.imshow(y_train[8,0][:, :, 0], cmap='gray')

    plt.tight_layout()
    plt.show()
    quit()
    """


def load_data(data_path, sequence_length, no_sequences=None):
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
            if one_file.split(".")[-1] != 'npy':
                write_to_summary(f"Skipping {root}/{one_file}", print_red=True)
                continue
            write_to_summary(f"Loading from:{root}/{one_file}")
            file_path = os.path.join(root, one_file)
            if all_data is False:
                all_data = load_blob(file_path)
                frames_available += all_data.shape[0]
            else:
                more_data = load_blob(file_path)
                all_data = np.concatenate((all_data, more_data), axis=0)
                frames_available += more_data.shape[0]
            # Add 10 sequences in case some are discarded for damaged frames
            if no_sequences is not None and frames_available // sequence_length > no_sequences + 10:
                break
    if all_data is False:
        return (False,False)
    # First we check the average, and see if there are any frames to discard
    average_pixel_count = [np.sum(frame) for frame in all_data]
    average_pixel_count = np.mean(average_pixel_count)
    write_to_summary(f"Average pixel count:{average_pixel_count}")
    skip_limit = DISCARD_CONSTANT * average_pixel_count
    write_to_summary(f"Pixel count skip limit:{skip_limit}")
    skip_indexes = []
    index_counter = 0
    for frame in all_data:
        if np.sum(frame) > skip_limit:
            skip_indexes.append(index_counter)
        index_counter += 1
    write_to_summary(f"{len(skip_indexes)} frames have a pixel count exceding the threshold:")
    write_to_summary(skip_indexes)

    # Generate all indicies that will produce data
    # and use that to filter out the ones with damaged frame in them 
    indicies_pairs = []
    frame_counter = 0
    while frame_counter + sequence_length + 1 < frames_available:
        all_valid_frames = True
        for damaged_frames in skip_indexes:
            if damaged_frames >= frame_counter and damaged_frames <= frame_counter + sequence_length + 1:
                all_valid_frames = False
                break
        pair = (frame_counter, frame_counter + sequence_length + 1)
        if not all_valid_frames:
            write_to_summary(f"{pair} skipped because of damaged frame", print_red=True)
        else:
            indicies_pairs.append(pair)
        frame_counter += sequence_length + 1
    write_to_summary(f"{len(indicies_pairs)} valid sequences available, target is {no_sequences}")

    # Check how many sequences we will get
    # final_no_sequences = frames_available // sequence_length
    final_no_sequences = len(indicies_pairs)
    if no_sequences is not None and final_no_sequences > no_sequences:
        final_no_sequences = no_sequences
        # Discard the ones we dont need
        indicies_pairs = indicies_pairs[:final_no_sequences]
    img_width = all_data.shape[1]
    img_height = all_data.shape[2]
    # -1 in sequence_length becasue the final frame is in the ground truth, no wait skip that
    # better to use sequence_length + 1 for y_train, makes more sense
    x_train = np.zeros((final_no_sequences, sequence_length, img_width, img_height, 1))
    y_train = np.zeros((final_no_sequences, 1, img_width, img_height, 1))
    current_sequence = 0
    for start_frame, end_frame in indicies_pairs:
        training_frames = all_data[start_frame: start_frame + sequence_length]
        truth_frame = all_data[start_frame + sequence_length: end_frame]
        x_train[current_sequence] = np.expand_dims(training_frames, axis=3)
        y_train[current_sequence] = np.expand_dims(truth_frame, axis=3)
        current_sequence += 1
    # No validation for now
    write_to_summary(f"Loaded {len(x_train)} sequences of length {sequence_length}!")
    return (x_train, y_train)

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

def write_to_summary(str_to_write, print_red=False, print_green=False):
    with open(SUMMARY_FILE, 'a') as f:
        f.write(f"{str_to_write}\n")
    cprint(str_to_write, print_red=print_red, print_green=print_green)

def cprint(string, print_red=False, print_green=False):
    if print_red:
        print(f'\033[91m{string}\033[0m')
    elif print_green:
        print(f'\033[92m{string}\033[0m')
    else:
        print(f'\033[93m{string}\033[0m')

if __name__=="__main__":
    main()
