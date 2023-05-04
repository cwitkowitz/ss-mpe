# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
#from NSynth import NSynthValidation
from Bach10 import Bach10
from Su import Su
#from TRIOS import TRIOS
#from MedleyDBPitch import MedleyDB_Pitch
#from MusicNet import MusicNet
#from URMP import URMP
#from SWD import SWD

from lhvqt import torch_amplitude_to_db
from hcqt import LHVQT

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


########################
## FEATURE EXTRACTION ##
########################

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
             to_db=False,
             db_to_prob=False,
             batch_norm=False)

# Create weighting for harmonics (harmonic loss)
harmonic_weights = 1 / torch.Tensor(harmonics) ** 2
# Apply zero weight to sub-harmonics (harmonic loss)
harmonic_weights[harmonic_weights > 1] = 0
# Normalize the harmonic weights
harmonic_weights /= torch.sum(harmonic_weights)
# Add frequency and time dimensions for broadcasting
harmonic_weights = harmonic_weights.unsqueeze(-1).unsqueeze(-1)


############
## MODELS ##
############

from basic_pitch.note_creation import model_frames_to_time
from basic_pitch import ICASSP_2022_MODEL_PATH
from basic_pitch.inference import predict
import tensorflow as tf

# Load the BasicPitch model checkpoint corresponding to paper
basic_pitch = tf.saved_model.load(str(ICASSP_2022_MODEL_PATH))
# Determine the MIDI frequency associated with each bin of Basic Pitch predictions
bp_midi_freqs = librosa.note_to_midi('A0') + np.arange(264) / (bins_per_octave / 12)

# Construct the path to the model checkpoint to evaluate
model_path = os.path.join(experiment_dir, 'models', f'model-{checkpoint}.pt')

# Load a checkpoint of the SS-MPE model
#ss_mpe = torch.load(model_path, map_location='cpu')
#ss_mpe.eval()

# Determine the MIDI frequency associated with each bin of predictions
ss_midi_freqs = fmin + np.arange(n_bins) / (bins_per_octave / 12)

# Determine which channel of the features corresponds to the first harmonic
h_idx = harmonics.index(1)


##############
## DATASETS ##
##############

# Instantiate Bach10 dataset for evaluation
bach10 = Bach10(base_dir=None,
                sample_rate=sample_rate)

# Instantiate Su dataset for evaluation
su = Su(base_dir=None,
        sample_rate=sample_rate)

"""
# Instantiate TRIOS dataset for evaluation
trios = TRIOS(base_dir=None,
              sample_rate=sample_rate)

# Instantiate MusicNet dataset for evaluation
musicnet = MusicNet(base_dir=None,
                    sample_rate=sample_rate)

# Instantiate URMP dataset for evaluation
urmp = URMP(base_dir=None,
            sample_rate=sample_rate)

# Instantiate SWD dataset for evaluation
swd = SWD(base_dir=None,
          sample_rate=sample_rate)

# Instantiate MedleyDB pitch-tracking subset for evaluation
medleydb = MedleyDB_Pitch(base_dir=None,
                          sample_rate=sample_rate)

#seed_everything(0)

# Instantiate NSynth dataset for validation
nsynthvalid = NSynthValidation(base_dir=None,
                               splits=['valid'],
                               #n_tracks=150,
                               #midi_range=[bp_midi_freqs[0], bp_midi_freqs[-1]],
                               sample_rate=sample_rate)
"""


################
## EVALUATION ##
################

# Construct the path to the directory under which to save results
#save_dir = os.path.join(experiment_dir, 'comparisons')

# Make sure the results directory exists
#os.makedirs(save_dir, exist_ok=True)

# Construct a path to a file for saving results
#save_path = os.path.join(save_dir, f'checkpoint-{checkpoint}.txt')
save_path = None

#if os.path.exists(save_path):
#    # Reset the file if it already exists
#    os.remove(save_path)

# Initialize a device pointer for loading the models
device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() and gpu_id is not None else 'cpu')

# Add feature extraction and model to the appropriate device
#hcqt, ss_mpe = hcqt.to(device), ss_mpe.to(device)
hcqt = hcqt.to(device)


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
for eval_set in [bach10, su]:
    # Initialize evaluators for all models
    bp_evaluator = MultipitchEvaluator()
    #ss_evaluator = MultipitchEvaluator()
    ln_evaluator = MultipitchEvaluator()
    lg_evaluator = MultipitchEvaluator()
    hm_ln_evaluator = MultipitchEvaluator()
    hm_lg_evaluator = MultipitchEvaluator()

    print_and_log(f'Results for {eval_set.name()}:', save_path)

    # Loop through all tracks in the test set
    for i, track in enumerate(tqdm(eval_set.tracks)):
        # Extract the audio for this track and add to appropriate device
        audio = eval_set.get_audio(track).to(device).unsqueeze(0)
        # Obtain the times associated with each frame of features
        times_est = hcqt.get_times(audio)

        if eval_set.has_frame_level_annotations():
            # Extract the ground-truth multi-pitch annotations for this track
            times_ref, multi_pitch_ref = eval_set.get_ground_truth(track)
        else:
            # Construct the ground-truth multi-pitch annotations for this track
            times_ref, multi_pitch_ref = eval_set.get_ground_truth(track, times_est)

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

        with torch.no_grad():
            # Obtain spectral features in decibels
            features_dec = torch_amplitude_to_db(hcqt(audio))
            # Convert decibels to linear gain between 0 and 1
            features_lin = decibels_to_amplitude(features_dec)
            # Scale decibels to be between 0 and 1
            features_log = rescale_decibels(features_dec)
            # Obtain predictions from the self-supervised model
            #ss_salience = torch.sigmoid(ss_mpe(features_log).squeeze())

        # Peak-pick and threshold salience to obtain binarized activations
        #ss_salience = np.round(filter_non_peaks(ss_salience.cpu().numpy()))
        # Convert the activations to frame-level multi-pitch estimates
        #ss_multi_pitch = eval_set.activations_to_multi_pitch(ss_salience, ss_midi_freqs)

        # Compute results for self-supervised predictions
        #ss_results = ss_evaluator.evaluate(times_est, ss_multi_pitch, times_ref, multi_pitch_ref)

        # Obtain salience as the first harmonic of the CQT features
        ln_salience = features_lin.squeeze()[h_idx].cpu().numpy()
        lg_salience = features_log.squeeze()[h_idx].cpu().numpy()
        hm_ln_salience = torch.sum(features_lin.squeeze() * harmonic_weights, dim=-3).cpu().numpy()
        hm_lg_salience = torch.sum(features_log.squeeze() * harmonic_weights, dim=-3).cpu().numpy()

        # Apply peak-picking to log-scale features
        lg_salience = filter_non_peaks(lg_salience)
        hm_lg_salience = filter_non_peaks(hm_lg_salience)

        # Convert saliences to frame-level multi-pitch estimates
        ln_multi_pitch = eval_set.activations_to_multi_pitch(ln_salience, ss_midi_freqs)
        lg_multi_pitch = eval_set.activations_to_multi_pitch(lg_salience, ss_midi_freqs)
        hm_ln_multi_pitch = eval_set.activations_to_multi_pitch(hm_ln_salience, ss_midi_freqs)
        hm_lg_multi_pitch = eval_set.activations_to_multi_pitch(hm_lg_salience, ss_midi_freqs)

        # Determine performance floor when using raw CQT features as predictions
        ln_results = ln_evaluator.evaluate(times_est, ln_multi_pitch, times_ref, multi_pitch_ref)
        lg_results = lg_evaluator.evaluate(times_est, lg_multi_pitch, times_ref, multi_pitch_ref)
        hm_ln_results = hm_ln_evaluator.evaluate(times_est, hm_ln_multi_pitch, times_ref, multi_pitch_ref)
        hm_lg_results = hm_lg_evaluator.evaluate(times_est, hm_lg_multi_pitch, times_ref, multi_pitch_ref)

        if verbose:
            # Print results for the individual track
            print_and_log(f'\tResults for track \'{track}\' ({eval_set.name()}):', save_path)
            print_and_log(f'\t- BasicPitch: {bp_results}', save_path)
            #print_and_log(f'\t- Self-Supervised: {ss_results}', save_path)
            print_and_log(f'\t- Amplitude CQT: {ln_results}', save_path)
            print_and_log(f'\t- Log-Scaled CQT: {lg_results}', save_path)
            print_and_log(f'\t- H-Weighted (Lin) CQT: {hm_ln_results}', save_path)
            print_and_log(f'\t- H-Weighted (Log) CQT: {hm_lg_results}', save_path)
            print_and_log('', save_path)

        # Track results with the respective evaluator
        bp_evaluator.append_results(bp_results)
        #ss_evaluator.append_results(ss_results)
        ln_evaluator.append_results(ln_results)
        lg_evaluator.append_results(lg_results)
        hm_ln_evaluator.append_results(hm_ln_results)
        hm_lg_evaluator.append_results(hm_lg_results)

    # Print average results for each evaluator
    print_and_log(f'BasicPitch: {bp_evaluator.average_results()[0]}', save_path)
    #print_and_log(f'Self-Supervised: {ss_evaluator.average_results()[0]}', save_path)
    print_and_log(f'Amplitude CQT: {ln_evaluator.average_results()[0]}', save_path)
    print_and_log(f'Log-Scaled CQT {lg_evaluator.average_results()[0]}', save_path)
    print_and_log(f'H-Weighted (Lin) CQT {hm_ln_evaluator.average_results()[0]}', save_path)
    print_and_log(f'H-Weighted (Log) CQT {hm_lg_evaluator.average_results()[0]}', save_path)
    print_and_log('', save_path)
