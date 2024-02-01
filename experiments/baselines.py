# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from timbre_trap.datasets.MixedMultiPitch import Bach10, URMP, Su, TRIOS
from timbre_trap.datasets.SoloMultiPitch import GuitarSet
from timbre_trap.datasets.NoteDataset import NoteDataset

from timbre_trap.datasets.utils import stream_url_resource, constants
from timbre_trap.models.utils import filter_non_peaks, threshold
from evaluate import MultipitchEvaluator
from ss_mpe.models import HCQT
from utils import *

# Regular imports
from tqdm import tqdm

import numpy as np
import mir_eval
import librosa
import torch
import os


# Choose the GPU on which to perform evaluation
gpu_id = None

# Flag to print results for each track separately
verbose = True

# File layout of system (0 - desktop | 1 - lab)
path_layout = 0


########################
## FEATURE EXTRACTION ##
########################

# Number of samples per second of audio
sample_rate = 22050

# Number of samples between frames
hop_length = 256

# First center frequency (MIDI) of geometric progression
fmin = librosa.note_to_midi('A0')

# Number of bins in a single octave
bins_per_octave = 60

# Number of frequency bins per CQT
n_bins = 440

# Harmonics to stack along channel dimension of HCQT
harmonics = [0.5, 1, 2, 3, 4, 5]

# Pack together HCQT parameters
hcqt_params = {'sample_rate': sample_rate,
               'hop_length': hop_length,
               'fmin': fmin,
               'bins_per_octave': bins_per_octave,
               'n_bins': n_bins,
               'gamma': None,
               'harmonics': harmonics}

# Initialize an HCQT module
hcqt = HCQT(**hcqt_params)


############
## MODELS ##
############

from basic_pitch.note_creation import model_frames_to_time
from basic_pitch import ICASSP_2022_MODEL_PATH
from basic_pitch.inference import predict
import tensorflow as tf

# Number of bins in a single octave
bp_bins_per_octave = 36
# Load the BasicPitch model checkpoint corresponding to paper
basic_pitch = tf.saved_model.load(str(ICASSP_2022_MODEL_PATH))
# Determine the MIDI frequency associated with each bin of Basic Pitch predictions
bp_midi_freqs = librosa.note_to_midi('A0') + np.arange(264) / (bp_bins_per_octave / 12)


# Specify the names of the files to download from GitHub
script_name, weights_name = 'predict_on_audio.py', 'multif0.h5'
# Obtain the path to the models directory
models_dir = os.path.join('..', 'ss_mpe', 'models')
# Construct a path to a top-level directory for DeepSalience
deep_salience_dir = os.path.join(models_dir, 'deep_salience')
# Create the necessary file hierarchy if it doesn't exist
os.makedirs(os.path.join(deep_salience_dir, 'weights'), exist_ok=True)
# Construct paths for the downloaded files
script_path = os.path.join(deep_salience_dir, script_name)
weights_path = os.path.join(deep_salience_dir, 'weights', weights_name)
try:
    # Attempt to import the DeepSalience inference code
    from ss_mpe.models.deep_salience.predict_on_audio import (model_def,
                                                              compute_hcqt,
                                                              get_single_test_prediction,
                                                              get_multif0)
except ModuleNotFoundError:
    # Point to the top-level directory containing files to download
    url_dir = 'https://raw.githubusercontent.com/rabitt/ismir2017-deepsalience/master/predict'
    # Construct the URLs of the files to download
    script_url, weights_url = f'{url_dir}/{script_name}', f'{url_dir}/weights/{weights_name}'
    # Download the script and weights files
    stream_url_resource(script_url, script_path)
    stream_url_resource(weights_url, weights_path)

    # Open the inference script for reading/writing
    with open(script_path, 'r+') as f:
        # Read all the code lines
        lines = f.readlines()  # Get a list of all lines
        # Reset file pointer
        f.seek(0)
        # Update lines of code
        lines[11] = 'from keras.layers import Input, Lambda, Conv2D, BatchNormalization\n'
        lines[69] = '\t\tBINS_PER_OCTAVE*N_OCTAVES, fmin=FMIN, bins_per_octave=BINS_PER_OCTAVE\n'
        # Remove outdated imports
        lines.pop(12)
        lines.pop(12)
        # Stop processing file
        f.truncate()
        # Overwrite the code
        f.writelines(lines)

    # Retry the original import
    from ss_mpe.models.deep_salience.predict_on_audio import (model_def,
                                                              compute_hcqt,
                                                              get_single_test_prediction,
                                                              get_multif0)
# Initialize DeepSalience
deep_salience = model_def()
# Load the weights from the paper
deep_salience.load_weights(weights_path)

# Initialize a device pointer for loading the models
device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() and gpu_id is not None else 'cpu')

# Construct the path to the final model checkpoint for the base Timbre-Trap model
model_path = os.path.join('..', 'generated', 'experiments', 'models', f'model-8750.pt')

# Load a checkpoint of the Timbre-Trap model
tt_mpe = torch.load(model_path, map_location=device)
tt_mpe.eval()

# TODO - add CREPE / PESTO


##############
## DATASETS ##
##############

# Point to the datasets within the storage drive containing them or use the default location
bch10_base_dir     = os.path.join('/', 'storage', 'frank', 'Bach10') if path_layout else None
urmp_base_dir      = os.path.join('/', 'storage', 'frank', 'URMP') if path_layout else None
su_base_dir        = os.path.join('/', 'storage', 'frank', 'Su') if path_layout else None
trios_base_dir     = os.path.join('/', 'storage', 'frank', 'TRIOS') if path_layout else None
gset_base_dir      = os.path.join('/', 'storage', 'frank', 'GuitarSet') if path_layout else None

# Instantiate Bach10 dataset mixtures for evaluation
bch10_test = Bach10(base_dir=bch10_base_dir,
                    splits=None,
                    sample_rate=sample_rate,
                    cqt=hcqt)

# Instantiate URMP dataset mixtures for evaluation
urmp_test = URMP(base_dir=urmp_base_dir,
                 splits=None,
                 sample_rate=sample_rate,
                 cqt=hcqt)

# Instantiate Su dataset for evaluation
su_test = Su(base_dir=su_base_dir,
             splits=None,
             sample_rate=sample_rate,
             cqt=hcqt)

# Instantiate TRIOS dataset for evaluation
trios_test = TRIOS(base_dir=trios_base_dir,
                   splits=None,
                   sample_rate=sample_rate,
                   cqt=hcqt)

# Instantiate GuitarSet dataset for evaluation
gset_test = GuitarSet(base_dir=gset_base_dir,
                      splits=None,
                      sample_rate=sample_rate,
                      cqt=hcqt)


################
## EVALUATION ##
################

# Construct the path to the directory under which to save baseline results
if path_layout == 1:
    save_dir = os.path.join('/', 'storage', 'frank', 'self-supervised-pitch')
else:
    save_dir = os.path.join('..', 'generated', 'experiments')

# Make sure the save directory exists
os.makedirs(save_dir, exist_ok=True)

# Construct a path to the file to save the comparison results
save_path = os.path.join(save_dir, f'baselines.txt')

if os.path.exists(save_path):
    # Reset the file if it already exists
    os.remove(save_path)

# Loop through validation and evaluation datasets
for eval_set in [bch10_test, urmp_test, su_test, trios_test, gset_test]:
    # Initialize evaluators for each algorithm/model
    bp_evaluator = MultipitchEvaluator()
    ds_evaluator = MultipitchEvaluator()
    tt_evaluator = MultipitchEvaluator()

    print_and_log(f'Results for {eval_set.name()}:', save_path)

    # Frequencies associated with ground-truth
    gt_midi_freqs = eval_set.cqt.get_midi_freqs()

    # Determine valid frequency bins for multi-pitch estimation (based on mir_eval)
    valid_freqs = librosa.midi_to_hz(gt_midi_freqs) > mir_eval.multipitch.MAX_FREQ

    # Loop through all tracks in the test set
    for i, data in enumerate(tqdm(eval_set)):
        # Determine which track is being processed
        track = data[constants.KEY_TRACK]

        if isinstance(eval_set, NoteDataset):
            # Extract frame times of ground-truth targets as reference
            times_ref = data[constants.KEY_TIMES]
            # Obtain the ground-truth note annotations
            pitches, intervals = eval_set.get_ground_truth(track)
            # Convert note pitches to Hertz
            pitches = librosa.midi_to_hz(pitches)
            # Convert the note annotations to multi-pitch annotations
            multi_pitch_ref = eval_set.notes_to_multi_pitch(pitches, intervals, times_ref)
        else:
            # Obtain the ground-truth multi-pitch annotations
            times_ref, multi_pitch_ref = eval_set.get_ground_truth(track)

        if verbose:
            # Print a header for the individual track's results
            print_and_log(f'\tResults for track \'{track}\' ({eval_set.name()}):', save_path)

        # Obtain a path for the track's audio
        audio_path = eval_set.get_audio_path(track)
        # Obtain predictions from the BasicPitch model
        model_output, _, _ = predict(audio_path, basic_pitch)
        # Extract the pitch salience predictions
        bp_salience = model_output['contour'].T
        # Determine times associated with each frame of predictions
        bp_times = model_frames_to_time(bp_salience.shape[-1])
        # Apply peak-picking and thresholding on the raw salience
        bp_salience = threshold(filter_non_peaks(bp_salience), 0.27)
        # Convert the activations to frame-level multi-pitch estimates
        bp_multi_pitch = eval_set.activations_to_multi_pitch(bp_salience, bp_midi_freqs)
        # Compute results for BasicPitch predictions
        bp_results = bp_evaluator.evaluate(bp_times, bp_multi_pitch, times_ref, multi_pitch_ref)
        # Store results for this track
        bp_evaluator.append_results(bp_results)

        if verbose:
            # Print results for the individual track
            print_and_log(f'\t\t-(bsc-ptc): {bp_results}', save_path)


        # Compute features for DeepSalience model
        hcqt, freq_grid, time_grid = compute_hcqt(audio_path)
        # Obtain predictions from the DeepSalience model
        ds_salience = get_single_test_prediction(deep_salience, hcqt)
        # Convert the activations to frame-level multi-pitch estimates
        ds_times, ds_multi_pitch = get_multif0(ds_salience, freq_grid, time_grid, thresh=0.3)
        # Compute results for DeepSalience predictions
        ds_results = ds_evaluator.evaluate(ds_times, ds_multi_pitch, times_ref, multi_pitch_ref)
        # Store results for this track
        ds_evaluator.append_results(ds_results)

        if verbose:
            # Print results for the individual track
            print_and_log(f'\t\t-(dp-slnc): {ds_results}', save_path)


        # Extract audio and add to the appropriate device
        audio = data[constants.KEY_AUDIO].to(device).unsqueeze(0)
        # Pad audio to next multiple of block length
        audio = tt_mpe.sliCQ.pad_to_block_length(audio)
        # Determine the times associated with features
        times_est = tt_mpe.sliCQ.get_times(tt_mpe.sliCQ.get_expected_frames(audio.size(-1)))
        # Transcribe the audio using the Timbre-Trap model
        tt_activations = to_array(tt_mpe.transcribe(audio).squeeze())
        # Peak-pick and threshold the Timbre-Trap activations
        tt_activations = threshold(filter_non_peaks(tt_activations), 0.5)
        # Remove activations for invalid frequencies
        tt_activations[valid_freqs] = 0
        # Convert the Timbre-Trap activations to frame-level multi-pitch estimates
        tt_multi_pitch = eval_set.activations_to_multi_pitch(tt_activations, gt_midi_freqs)
        # Compute results for predictions from the Timbre-Trap methodology
        tt_results = tt_evaluator.evaluate(times_est, tt_multi_pitch, times_ref, multi_pitch_ref)
        # Store results for this track
        tt_evaluator.append_results(tt_results)

        if verbose:
            # Print results for the individual track
            print_and_log(f'\t\t-(tt-mpe): {tt_results}', save_path)

    # Print a header for average results across all tracks of the dataset
    print_and_log(f'\tAverage Results ({eval_set.name()}):', save_path)

    # Print average results
    print_and_log(f'\t\t-(bsc-ptc): {bp_evaluator.average_results()[0]}', save_path)
    print_and_log(f'\t\t-(dp-slnc): {ds_evaluator.average_results()[0]}', save_path)
    print_and_log(f'\t\t-(tt-mpe): {tt_evaluator.average_results()[0]}', save_path)
