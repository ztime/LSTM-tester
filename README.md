# Keras LSTM

This is a "framework" for working with Tensorflow/Keras and training different models.

The idea is that you define you model in ``models/`` folder and then load it from ``lstm-train.py`` or ``lstm-predict.py``,
and then the script will automatically name things so that it's easier to keep track of what you've trained.

Using ``--save_tensorboard`` is a good idea, but also requires a bit more memory, so be careful.

A suggestion is using a bash script or something to run your models and save settings!
There should be one available on the ``rabbit`` server (the folder was called ``dcgan`` then) called ``run_several.sh`` or 
something similar.



```
usage: lstm-train.py [-h] -o OUTPUT -p PREFIX -m MODELFILE -e EPOCHS -d
                    DATAPATH -b BATCHSIZE [--sequence_length SEQUENCE_LENGTH]
                    [--no_of_sequences NO_OF_SEQUENCES] [--save_tensorboard]
                    [--validation_split VALIDATION_SPLIT]
                    [--save_metrics_each_batch SAVE_METRICS_EACH_BATCH]
                    [--save_plot_model] [--yes_to_all] [--use_moving_mnist]

optional arguments:
    -h, --help            show this help message and exit
    -o OUTPUT, --output OUTPUT
        Where to store logs/models
    -p PREFIX, --prefix PREFIX
        Prefix for naming all files
    -m MODELFILE, --modelfile MODELFILE
        What model to use
    -e EPOCHS, --epochs EPOCHS
        Number of epochs to train
    -d DATAPATH, --datapath DATAPATH
        Path to training data
    -b BATCHSIZE, --batchsize BATCHSIZE
        What batchsize to use
    --sequence_length SEQUENCE_LENGTH
        How long sequence length in lstm
    --no_of_sequences NO_OF_SEQUENCES
        How many sequences to load, default is all
    --save_tensorboard    Save tensorboard logs
    --validation_split VALIDATION_SPLIT
        Split training data into validation ratio
    --save_metrics_each_batch SAVE_METRICS_EACH_BATCH
        Save metrics after each batch to file
    --save_plot_model     Save an image plot of the model
     --yes_to_all          Answer yes to all questions!
    --use_moving_mnist    Use the moving mnist dataset (overrides)

```

```
usage: lstm-predict.py [-h] -o OUTPUT -p PREFIX -m MODEL -s
                        {zeros,ones,random,real} -f FRAMES
                        [--real_seed_path REAL_SEED_PATH]
                        [--real_seed_offset REAL_SEED_OFFSET]
                        [--save_numpy_file] [--save_images] [--save_video]
                        [--save_zip] [--framerate FRAMERATE] [-y]
                        [--use_moving_mnist] [--use_output_no USE_OUTPUT_NO]
                        [--clip_frames]

optional arguments:
    -h, --help            show this help message and exit
    -o OUTPUT, --output OUTPUT
        Folder to store results in
    -p PREFIX, --prefix PREFIX
        Prefix for the files
    -m MODEL, --model MODEL
        What model to generate from (hdf5 file)
    -s {zeros,ones,random,real}, --seed_data {zeros,ones,random,real}
        What seed data to use
    -f FRAMES, --frames FRAMES
        How many frames to generate (full length will be
        sequence_length + frames)
    --real_seed_path REAL_SEED_PATH
        Path to seed data to use, required if seed_data = real
    --real_seed_offset REAL_SEED_OFFSET
        Offset in the real seed file to start sampling from
    --save_numpy_file     Store all frames as numpy file
    --save_images         Store all images in result folder
    --save_video          Store an video in results folder
    --save_zip            Zip the result-folder for easy transfer
    --framerate FRAMERATE
        Framerate for generating video, not to be confused
        with frames
    -y, --yes_to_all
    --use_moving_mnist
    --use_output_no USE_OUTPUT_NO
        If model has several outputs, use this index
    --clip_frames         Clip frames with .5 > = 1 and < .5 = 0

```
