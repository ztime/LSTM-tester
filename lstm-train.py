import argparse
import os
import importlib
import numpy as np
import datetime

from keras.callbacks import ModelCheckpoint, TensorBoard, Callback

from pprint import pprint

SUMMARY_FILE = None

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
    parser.add_argument('--save_tensorboard', action='store_true', default=True, help='Save tensorboard logs')
    parser.add_argument('--validation_split', type=float, default=0.2, help="Split training data into validation ratio")
    parser.add_argument('--save_metrics_each_batch', help='Save metrics after each batch to file')

    args = parser.parse_args()

    # Set up summary file
    date_separator = datetime.datetime.now().strftime("%d%m%Y-%H%M%S")
    summary_file_name = f"{args.prefix}-summary-{date_separator}.log"
    SUMMARY_FILE = os.path.join(args.output, summary_file_name)

    # Check folder
    if os.path.isdir(args.output):
        y_n = input(f"'{args.output}' already exists, continue anyway?[y/N]")
        if y_n != 'y':
            quit()
    else:
        os.mkdir(args.output)

    # Check datapath, load data bc we need to know image dimensions of the trainingdata
    if args.no_of_sequences == -1:
        args.no_of_sequences = None
    x_train, y_train = load_data(args.datapath, args.sequence_length, args.no_of_sequences)
    if x_train is False or y_train is False:
        cprint("Could not load data! Aborting...", print_red=True)
        quit()
    img_width, img_height = x_train[0].shape[1], x_train[0].shape[2]

    write_to_summary("Loaded data, folder created.")
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
    model = loaded_model.get_model(args.sequence_length, img_width, img_height)
    desc = loaded_model.get_description()
    # We can let the model override some parameters
    if hasattr(loaded_model, 'MODEL_OVERRIDES'):
        for param_name, param_value in loaded_model.MODEL_OVERRIDES.items():
            if param_name == 'sequence_length':
                cprint(f"FYI: Model override the sequence length from {args.sequence_length} to {param_value}", print_red=True)
                args.sequence_length = param_value
            elif param_name == 'batch_size':
                cprint(f"FYI: Model override batchsize from {args.batchsize} to {param_value}", print_red=True)
                args.batchsize = param_value
            else:
                # Fallback
                cprint(f"Warning, model tried to override {param_name} but that is not supported!", print_red=True)

    write_to_summary("Loaded model successfully!", print_green=True)
    write_to_summary("Summary:")
    model.summary(print_fn=lambda x: write_to_summary(x))
    write_to_summary("Description:")
    write_to_summary(desc)
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
        tensorboard_callback = TensorBoard(log_dir=tensorboard_filepath, histogram_freq=0, write_graph=True, write_images=True)
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
    try:
        history = model.fit(
                x_train,
                y_train,
                batch_size=args.batchsize,
                epochs=args.epochs,
                validation_split=args.validation_split,
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
    model_filename = os.path.join(args.output, 'saved_models', 'model--last--saved.hdf5')
    model.save(model_filename)
    


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
            write_to_summary(f"Loading from:{root}/{one_file}")
            file_path = os.path.join(root, one_file)
            if all_data is False:
                all_data = load_blob(file_path)
                frames_available += all_data.shape[0]
            else:
                more_data = load_blob(file_path)
                all_data = np.concatenate((all_data, more_data), axis=0)
                frames_available += more_data.shape[0]
            if no_sequences is not None and frames_available // sequence_length > no_sequences:
                break
    if all_data is False:
        return (False,False)
    # Check how many sequences we will get
    final_no_sequences = frames_available // sequence_length
    if no_sequences is not None and final_no_sequences > no_sequences:
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
        if current_sequence >= final_no_sequences - 1:
            break
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
