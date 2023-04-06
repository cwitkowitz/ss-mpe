# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from NSynth import NSynth
from Bach10 import Bach10
from Su import Su
from TRIOS import TRIOS
from MusicNet import MusicNet
from lhvqt import LHVQT
from evaluate import MultipitchEvaluator
from utils import *

# Regular imports
from tqdm import tqdm

import librosa
import torch
import os


# Choose the GPU on which to perform evaluation
gpu_id = None

# Choose the model checkpoint to compare
checkpoint = 0

# Construct the path to the top-level directory of the experiment
experiment_dir = os.path.join('.', 'generated', 'experiments', '<EXPERIMENT_DIR>')

# Flag to print results for each track separately
verbose = False

##############################
## FEATURE EXTRACTION

# Number of samples per second of audio
sample_rate = 22050

# Number of samples between frames
hop_length = 512

# Number of frequency bins per CQT
n_bins = 216

# Number of bins in a single octave
bins_per_octave = 36

# Harmonics to stack along channel dimension of HCQT
harmonics = [0.5, 1, 2, 3, 4, 5]

# First center frequency (MIDI) of geometric progression
fmin = librosa.note_to_midi('C1')

# Initialize the HCQT feature extraction module
hcqt = LHVQT(fs=sample_rate,
             hop_length=hop_length,
             fmin=librosa.midi_to_hz(fmin),
             n_bins=n_bins,
             bins_per_octave=bins_per_octave,
             harmonics=harmonics,
             update=False,
             db_to_prob=False,
             batch_norm=False)

##############################
## MODELS

#from basic_pitch.inference import predict
#from basic_pitch import ICASSP_2022_MODEL_PATH

#import tensorflow as tf

# Load the BasicPitch model checkpoint corresponding to paper
#basic_pitch = tf.saved_model.load(str(ICASSP_2022_MODEL_PATH))

# Construct the path to the model checkpoint to evaluate
model_path = os.path.join(experiment_dir, 'models', f'model-{checkpoint}.pt')

# Load a checkpoint of the SS-MPE model
ss_mpe = torch.load(model_path, map_location='cpu')
ss_mpe.eval()

# Determine which channel of the features corresponds to the first harmonic
h_idx = harmonics.index(1)

##############################
## DATASETS

# Instantiate Bach10 dataset for validation
bach10 = Bach10(sample_rate=sample_rate,
                hop_length=hop_length,
                fmin=fmin,
                n_bins=n_bins,
                bins_per_octave=bins_per_octave)

# Instantiate Su dataset for validation
su = Su(sample_rate=sample_rate,
        hop_length=hop_length,
        fmin=fmin,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave)

# Instantiate Su dataset for validation
trios = TRIOS(sample_rate=sample_rate,
              hop_length=hop_length,
              fmin=fmin,
              n_bins=n_bins,
              bins_per_octave=bins_per_octave)

# Instantiate Su dataset for validation
musicnet = MusicNet(sample_rate=sample_rate,
                    hop_length=hop_length,
                    fmin=fmin,
                    n_bins=n_bins,
                    bins_per_octave=bins_per_octave)

##############################
## EVALUATION

# Construct the path to the directory under which to save results
save_dir = os.path.join(experiment_dir, 'comparisons')

# Make sure the results directory exists
os.makedirs(save_dir, exist_ok=True)

# Construct a path to a file for saving results
save_path = os.path.join(save_dir, f'checkpoint-{checkpoint}.txt')

if os.path.exists(save_path):
    # Reset the file if it already exists
    os.remove(save_path)

# Initialize a device pointer for loading the models
device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() and gpu_id is not None else 'cpu')

# Add feature extraction and model to the appropriate device
hcqt, ss_mpe = hcqt.to(device), ss_mpe.to(device)


def print_and_log(text, path=None):
    """
    TODO
    """

    # Print text to the console
    print(text)

    if path is not None:
        with open(path, 'a') as f:
            # Append the text to the file
            print(text, file=f)


# Loop through all evaluation datasets
for test_set in [bach10, su, trios, musicnet]:
    # Initialize evaluators for all models
    bp_evaluator = MultipitchEvaluator()
    ss_evaluator = MultipitchEvaluator()
    ln_evaluator = MultipitchEvaluator()
    sc_evaluator = MultipitchEvaluator()

    print_and_log(f'Results for {test_set.name()}:', save_path)

    # Loop through all tracks in the test set
    for i, track in enumerate(tqdm(test_set.tracks)):
        # Obtain a path for the track's audio
        audio_path = test_set.get_audio_path(track)
        # Extract the audio and ground-truth
        audio, ground_truth = test_set[i]

        # Add audio to the appropriate device
        audio = audio.to(device)

        # Convert ground-truth to NumPy array
        ground_truth = ground_truth.numpy()

        # Obtain predictions from the BasicPitch model
        #model_output, _, _ = predict(audio_path, basic_pitch)
        #bp_salience = model_output['contour'].T
        # TODO - resample output to match ground-truth

        # Compute results for BasicPitch predictions
        #bp_results = bp_evaluator.evaluate(bp_salience, ground_truth)

        with torch.no_grad():
            # Obtain features for the audio
            features = hcqt(audio.unsqueeze(0))
            features_l = decibels_to_amplitude(features)
            features_s = rescale_decibels(features)
            # Obtain predictions from the self-supervised model
            ss_salience = torch.sigmoid(ss_mpe(features_s).squeeze())

        # Compute results for self-supervised predictions
        ss_results = ss_evaluator.evaluate(ss_salience.cpu().numpy(), ground_truth)

        # Obtain salience as the first harmonic of the CQT features
        ln_salience = features_l.squeeze()[h_idx]
        #ln_salience = features_l.squeeze().mean(0)
        sc_salience = features_s.squeeze()[h_idx]
        #sc_salience = features_s.squeeze().mean(0)

        # Determine performance floor when using CQT features as predictions
        ln_results = ln_evaluator.evaluate(ln_salience.cpu().numpy(), ground_truth)
        sc_results = sc_evaluator.evaluate(sc_salience.cpu().numpy(), ground_truth)

        if verbose:
            # Print results for the individual track
            print_and_log(f'\tResults for track \'{track}\' ({test_set.name()}):', save_path)
            #print_and_log(f'\t- BasicPitch: {bp_results}', save_path)
            print_and_log(f'\t- Self-Supervised: {ss_results}', save_path)
            print_and_log(f'\t- Amplitude CQT: {ln_results}', save_path)
            print_and_log(f'\t- Log-Scaled CQT: {sc_results}', save_path)
            print_and_log('', save_path)

        # Track results with the respective evaluator
        #bp_evaluator.append_results(bp_results)
        ss_evaluator.append_results(ss_results)
        ln_evaluator.append_results(ln_results)
        sc_evaluator.append_results(sc_results)

    # Print average results for each evaluator
    #print_and_log('BasicPitch: {bp_evaluator.average_results()}', save_path)
    print_and_log(f'Self-Supervised: {ss_evaluator.average_results()}', save_path)
    print_and_log(f'Amplitude CQT: {ln_evaluator.average_results()}', save_path)
    print_and_log(f'Log-Scaled CQT {sc_evaluator.average_results()}', save_path)
    print_and_log('', save_path)
