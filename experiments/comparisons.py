# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from timbre_trap.datasets.MixedMultiPitch import URMP, Bach10, Su, TRIOS
from timbre_trap.datasets.SoloMultiPitch import GuitarSet
from timbre_trap.datasets.NoteDataset import NoteDataset

from timbre_trap.datasets.utils import stream_url_resource, constants
from timbre_trap.models.utils import filter_non_peaks, threshold
from evaluate import MultipitchEvaluator
from ss_mpe.models import SS_MPE
from utils import *

# Regular imports
from tqdm import tqdm

import numpy as np
import librosa
import torch
import os


# Name of the model to evaluate
ex_name = '<EXPERIMENT_DIR>'

# Choose the model checkpoint to compare
checkpoint = 0

# Choose the GPU on which to perform evaluation
gpu_id = None

# Flag to print results for each track separately
#verbose = True
verbose = False

# File layout of system (0 - desktop | 1 - lab)
path_layout = 0

# Construct the path to the top-level directory of the experiment
if path_layout == 1:
    experiment_dir = os.path.join('/', 'storage', 'frank', 'self-supervised-pitch', ex_name)
else:
    experiment_dir = os.path.join('..', 'generated', 'experiments', ex_name)


########################
## FEATURE EXTRACTION ##
########################

# Number of samples per second of audio
sample_rate = 22050


############
## MODELS ##
############

# Initialize a device pointer for loading the models
device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() and gpu_id is not None else 'cpu')

# Construct the path to the model checkpoint to evaluate
model_path = os.path.join(experiment_dir, 'models', f'model-{checkpoint}.pt')

# Load a checkpoint of the SS-MPE model
ss_mpe = SS_MPE.load(model_path, device=device)
ss_mpe.eval()


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


# TODO - include models from https://ieeexplore.ieee.org/abstract/document/9865174?


##############
## DATASETS ##
##############

# Point to the datasets within the storage drive containing them or use the default location
urmp_base_dir  = os.path.join('/', 'storage', 'frank', 'URMP') if path_layout else None
bch10_base_dir = os.path.join('/', 'storage', 'frank', 'Bach10') if path_layout else None
gset_base_dir  = os.path.join('/', 'storage', 'frank', 'GuitarSet') if path_layout else None
su_base_dir    = os.path.join('/', 'storage', 'frank', 'Su') if path_layout else None
trios_base_dir = os.path.join('/', 'storage', 'frank', 'TRIOS') if path_layout else None

# Set the URMP validation set as was defined in the MT3 paper
urmp_val_splits = ['01', '02', '12', '13', '24', '25', '31', '38', '39']
# Instantiate URMP dataset mixtures for evaluation
urmp_test = URMP(base_dir=urmp_base_dir,
                 splits=urmp_val_splits,
                 sample_rate=sample_rate,
                 cqt=ss_mpe.hcqt)

# Instantiate TRIOS dataset for evaluation
trios_test = TRIOS(base_dir=trios_base_dir,
                   splits=None,
                   sample_rate=sample_rate,
                   cqt=ss_mpe.hcqt)

# Instantiate Bach10 dataset mixtures for evaluation
bch10_test = Bach10(base_dir=bch10_base_dir,
                    splits=None,
                    sample_rate=sample_rate,
                    cqt=ss_mpe.hcqt)

# Instantiate Su dataset for evaluation
su_test = Su(base_dir=su_base_dir,
             splits=None,
             sample_rate=sample_rate,
             cqt=ss_mpe.hcqt)

# Instantiate GuitarSet dataset for evaluation
gset_test = GuitarSet(base_dir=gset_base_dir,
                      splits=['05'],
                      sample_rate=sample_rate,
                      cqt=ss_mpe.hcqt,)


################
## EVALUATION ##
################

# Construct a path to the directory under which to save comparisons
save_dir = os.path.join(experiment_dir, 'comparisons')

# Make sure the comparison directory exists
os.makedirs(save_dir, exist_ok=True)

# Construct a path to the file to save the comparison results
save_path = os.path.join(save_dir, f'checkpoint-{checkpoint}.txt')

if os.path.exists(save_path):
    # Reset the file if it already exists
    os.remove(save_path)

# Loop through validation and evaluation datasets
for eval_set in [urmp_test, trios_test, bch10_test, su_test, gset_test]:
    # Initialize evaluators for each algorithm/model
    ln1_evaluator = MultipitchEvaluator()
    lg1_evaluator = MultipitchEvaluator()
    lnh_evaluator = MultipitchEvaluator()
    lgh_evaluator = MultipitchEvaluator()
    ss_evaluator = MultipitchEvaluator()
    bp_evaluator = MultipitchEvaluator()
    ds_evaluator = MultipitchEvaluator()

    print_and_log(f'Results for {eval_set.name()}:', save_path)

    # Frequencies associated with ground-truth
    gt_midi_freqs = eval_set.cqt.midi_freqs

    # Loop through all tracks in the test set
    for i, data in enumerate(tqdm(eval_set)):
        # Determine which track is being processed
        track = data[constants.KEY_TRACK]
        # Extract audio and add to the appropriate device
        audio = data[constants.KEY_AUDIO].to(device).unsqueeze(0)

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

        # Determine the times associated with predictions
        times_est = eval_set.cqt.get_times(eval_set.cqt.get_expected_frames(audio.size(-1)))

        # Compute full set of spectral features
        features = ss_mpe.get_all_features(audio)

        # Extract relevant feature sets
        features_lin_1 = features['amp_1']
        features_log_1 = features['dec_1']
        features_lin_h = features['amp_h']
        features_log_h = features['dec_h']

        # Peak-pick and threshold the linear-scaled magnitude
        ln1_activations = threshold(filter_non_peaks(to_array(features_lin_1)), 0.3)
        # Convert the raw-feature activations to frame-level multi-pitch estimates
        ln1_multi_pitch = eval_set.activations_to_multi_pitch(ln1_activations, gt_midi_freqs)
        # Compute results for predictions from the linear-scaled CQT features
        ln1_results = ln1_evaluator.evaluate(times_est, ln1_multi_pitch, times_ref, multi_pitch_ref)
        # Store results for this track
        ln1_evaluator.append_results(ln1_results)

        if verbose:
            # Print results for the individual track
            print_and_log(f'\t\t-(lin-cqt-1): {ln1_results}', save_path)


        # Peak-pick and threshold the log-scaled magnitude
        lg1_activations = threshold(filter_non_peaks(to_array(features_log_1)), 0.8)
        # Convert the raw-feature activations to frame-level multi-pitch estimates
        lg1_multi_pitch = eval_set.activations_to_multi_pitch(lg1_activations, gt_midi_freqs)
        # Compute results for predictions from the log-scaled CQT features
        lg1_results = lg1_evaluator.evaluate(times_est, lg1_multi_pitch, times_ref, multi_pitch_ref)
        # Store results for this track
        lg1_evaluator.append_results(lg1_results)

        if verbose:
            # Print results for the individual track
            print_and_log(f'\t\t-(log-cqt-1): {lg1_results}', save_path)

        # Peak-pick and threshold the weighted harmonic average linear-scaled magnitude
        lnh_activations = threshold(filter_non_peaks(to_array(features_lin_h)), 0.3)
        # Convert the raw-feature activations to frame-level multi-pitch estimates
        lnh_multi_pitch = eval_set.activations_to_multi_pitch(lnh_activations, gt_midi_freqs)
        # Compute results for predictions from the linear-scaled CQT features
        lnh_results = lnh_evaluator.evaluate(times_est, lnh_multi_pitch, times_ref, multi_pitch_ref)
        # Store results for this track
        lnh_evaluator.append_results(lnh_results)

        if verbose:
            # Print results for the individual track
            print_and_log(f'\t\t-(lin-cqt-h): {lnh_results}', save_path)


        # Peak-pick and threshold the weighted harmonic average log-scaled magnitude
        lgh_activations = threshold(filter_non_peaks(to_array(features_log_h)), 0.5)
        # Convert the raw-feature activations to frame-level multi-pitch estimates
        lgh_multi_pitch = eval_set.activations_to_multi_pitch(lgh_activations, gt_midi_freqs)
        # Compute results for predictions from the log-scaled CQT features
        lgh_results = lgh_evaluator.evaluate(times_est, lgh_multi_pitch, times_ref, multi_pitch_ref)
        # Store results for this track
        lgh_evaluator.append_results(lgh_results)

        if verbose:
            # Print results for the individual track
            print_and_log(f'\t\t-(log-cqt-h): {lgh_results}', save_path)


        # Transcribe the audio using the SS-MPE model
        ss_activations = to_array(ss_mpe.transcribe(audio).squeeze())
        # Peak-pick and threshold the SS-MPE activations
        ss_activations = threshold(filter_non_peaks(ss_activations), 0.5)
        # Convert the SS-MPE activations to frame-level multi-pitch estimates
        ss_multi_pitch = eval_set.activations_to_multi_pitch(ss_activations, gt_midi_freqs)
        # Compute results for predictions from the SS-MPE methodology
        ss_results = ss_evaluator.evaluate(times_est, ss_multi_pitch, times_ref, multi_pitch_ref)
        # Store results for this track
        ss_evaluator.append_results(ss_results)

        if verbose:
            # Print results for the individual track
            print_and_log(f'\t\t-(ss-mpe): {ss_results}', save_path)


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

    # Print a header for average results across all tracks of the dataset
    print_and_log(f'\tAverage Results ({eval_set.name()}):', save_path)

    # Print average results
    print_and_log(f'\t\t-(lin-cqt-1): {ln1_evaluator.average_results()[0]}', save_path)
    print_and_log(f'\t\t-(log-cqt-1): {lg1_evaluator.average_results()[0]}', save_path)
    print_and_log(f'\t\t-(lin-cqt-h): {lnh_evaluator.average_results()[0]}', save_path)
    print_and_log(f'\t\t-(log-cqt-h): {lgh_evaluator.average_results()[0]}', save_path)
    print_and_log(f'\t\t-(ss-mpe): {ss_evaluator.average_results()[0]}', save_path)
    print_and_log(f'\t\t-(bsc-ptc): {bp_evaluator.average_results()[0]}', save_path)
    print_and_log(f'\t\t-(dp-slnc): {ds_evaluator.average_results()[0]}', save_path)
