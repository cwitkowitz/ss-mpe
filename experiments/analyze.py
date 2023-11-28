# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from timbre_trap.datasets.NoteDataset import NoteDataset
from ss_mpe.datasets.SoloMultiPitch import NSynth

from timbre_trap.models.utils import filter_non_peaks, threshold
from timbre_trap.datasets.utils import constants
from evaluate import MultipitchEvaluator
from ss_mpe.models import SS_MPE
from utils import *

# Regular imports
from tqdm import tqdm

import matplotlib.pyplot as plt
import librosa
import torch
import os


# Name of the model to evaluate
ex_name = 'Timbre_Gaus_NW_LR5e-4_2'
#ex_name = '<EXPERIMENT_DIR>'

# Choose the model checkpoint to compare
checkpoint = 21500
#checkpoint = 0

# Choose the GPU on which to perform evaluation
gpu_id = None

# Flag to print results for each track separately
verbose = True

# File layout of system (0 - desktop | 1 - lab)
path_layout = 0

# Construct the path to the top-level directory of the experiment
if path_layout == 1:
    experiment_dir = os.path.join('/', 'storage', 'frank', 'self-supervised-pitch', ex_name)
else:
    experiment_dir = os.path.join('..', 'generated', 'experiments', ex_name)

# Set randomization seed
seed = 0

# Seed everything with the same seed
seed_everything(seed)


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


##############
## DATASETS ##
##############

# Point to the datasets within the storage drive containing them or use the default location
nsynth_base_dir = os.path.join('/', 'storageNVME', 'frank', 'NSynth') if path_layout else None

# Instantiate NSynth validation split for validation
nsynth_val = NSynth(base_dir=nsynth_base_dir,
                    splits=['valid'],
                    n_tracks=200,
                    sample_rate=sample_rate,
                    cqt=ss_mpe.hcqt,
                    seed=seed)

# Sample from which to start NSynth analysis
nsynth_start_idx = 0

# Slice NSynth dataset
nsynth_val.tracks = nsynth_val.tracks[nsynth_start_idx:]


################
## EVALUATION ##
################

# Loop through validation and evaluation datasets
for eval_set in [nsynth_val]:
    # Initialize evaluators for each algorithm/model
    ln1_evaluator = MultipitchEvaluator()
    lg1_evaluator = MultipitchEvaluator()
    lnh_evaluator = MultipitchEvaluator()
    lgh_evaluator = MultipitchEvaluator()
    ss_evaluator = MultipitchEvaluator()

    print(f'Results for {eval_set.name()}:')

    # Frequencies associated with ground-truth
    gt_midi_freqs = eval_set.cqt.get_midi_freqs()

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
            print(f'\tResults for track \'{track}\' ({eval_set.name()}):')

        # Determine the times associated with predictions
        times_est = eval_set.cqt.get_times(eval_set.cqt.get_expected_frames(audio.size(-1)))

        # Compute full set of spectral features
        features = ss_mpe.get_all_features(audio)

        # Extract relevant feature sets
        features_lin_1 = features['amp_1']
        features_log_1 = features['dec_1']
        features_lin_h = features['amp_h']
        features_log_h = features['dec_h']

        # Extract ground-truth pitch salience activations
        gt_activations = data[constants.KEY_GROUND_TRUTH]

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
            print(f'\t\t-(lin-cqt-1): {ln1_results}')


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
            print(f'\t\t-(log-cqt-1): {lg1_results}')

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
            print(f'\t\t-(lin-cqt-h): {lnh_results}')


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
            print(f'\t\t-(log-cqt-h): {lgh_results}')


        # Transcribe the audio using the SS-MPE model
        ss_activations_ = to_array(ss_mpe.transcribe(audio).squeeze())
        # Peak-pick and threshold the SS-MPE activations
        ss_activations = threshold(filter_non_peaks(ss_activations_), 0.5)
        # Convert the SS-MPE activations to frame-level multi-pitch estimates
        ss_multi_pitch = eval_set.activations_to_multi_pitch(ss_activations, gt_midi_freqs)
        # Compute results for predictions from the SS-MPE methodology
        ss_results = ss_evaluator.evaluate(times_est, ss_multi_pitch, times_ref, multi_pitch_ref)
        # Store results for this track
        ss_evaluator.append_results(ss_results)

        if verbose:
            # Print results for the individual track
            print(f'\t\t-(ss-mpe): {ss_results}')

        if verbose:
            # Initialize a new figure with subplots if one was not given
            (fig, ax) = plt.subplots(nrows=3, ncols=2, figsize=(8, 12))

            # Determine track's attributes
            name, pitch, vel = track.split('-')
            # Add a global title above all sub-plots
            fig.suptitle(f'Track: {name} | Pitch: {pitch} | Velocity: {vel}')

            # Plot ground-truth activations
            fig.sca(ax[0, 1])
            plot_magnitude(gt_activations, fig=fig)
            # Add subtitle
            ax[0, 1].axis('on')
            ax[0, 1].set_xticks([])
            ax[0, 1].get_yaxis().set_visible(False)
            ax[0, 1].set_xlabel('Ground-Truth')

            # Plot SS-MPE activations
            fig.sca(ax[0, 0])
            plot_magnitude(ss_activations_, fig=fig)
            # Extract SS-MPE performance measures
            ss_mpe_pr = ss_results['mpe/precision'].round(3)
            ss_mpe_rc = ss_results['mpe/recall'].round(3)
            ss_mpe_f1 = ss_results['mpe/f1-score'].round(3)
            # Add subtitle
            ax[0, 0].axis('on')
            ax[0, 0].set_xticks([])
            ax[0, 0].get_yaxis().set_visible(False)
            ax[0, 0].set_xlabel(f'SS-MPE - P: {ss_mpe_pr} | R: {ss_mpe_rc} | F1: {ss_mpe_f1}')

            # Plot 1st harmonic (log) activations
            fig.sca(ax[1, 0])
            plot_magnitude(features_log_1[0], fig=fig)
            # Extract 1st harmonic (log) performance measures
            lg1_mpe_pr = lg1_results['mpe/precision'].round(3)
            lg1_mpe_rc = lg1_results['mpe/recall'].round(3)
            lg1_mpe_f1 = lg1_results['mpe/f1-score'].round(3)
            # Add subtitle
            ax[1, 0].axis('on')
            ax[1, 0].set_xticks([])
            ax[1, 0].get_yaxis().set_visible(False)
            ax[1, 0].set_xlabel(f'Log 1st - P: {lg1_mpe_pr} | R: {lg1_mpe_rc} | F1: {lg1_mpe_f1}')

            # Plot 1st harmonic (linear) activations
            fig.sca(ax[1, 1])
            plot_magnitude(features_lin_1[0], fig=fig)
            # Extract 1st harmonic (linear) performance measures
            ln1_mpe_pr = ln1_results['mpe/precision'].round(3)
            ln1_mpe_rc = ln1_results['mpe/recall'].round(3)
            ln1_mpe_f1 = ln1_results['mpe/f1-score'].round(3)
            # Add subtitle
            ax[1, 1].axis('on')
            ax[1, 1].set_xticks([])
            ax[1, 1].get_yaxis().set_visible(False)
            ax[1, 1].set_xlabel(f'Lin. 1st - P: {ln1_mpe_pr} | R: {ln1_mpe_rc} | F1: {ln1_mpe_f1}')

            # Plot harmonic average (log) activations
            fig.sca(ax[2, 0])
            plot_magnitude(features_log_h[0], fig=fig)
            # Extract harmonic average (log) performance measures
            lgh_mpe_pr = lgh_results['mpe/precision'].round(3)
            lgh_mpe_rc = lgh_results['mpe/recall'].round(3)
            lgh_mpe_f1 = lgh_results['mpe/f1-score'].round(3)
            # Add subtitle
            ax[2, 0].axis('on')
            ax[2, 0].set_xticks([])
            ax[2, 0].get_yaxis().set_visible(False)
            ax[2, 0].set_xlabel(f'Log H.Avg. - P: {lgh_mpe_pr} | R: {lgh_mpe_rc} | F1: {lgh_mpe_f1}')

            # Plot harmonic average (linear) activations
            fig.sca(ax[2, 1])
            plot_magnitude(features_lin_h[0], fig=fig)
            # Extract harmonic average (linear) performance measures
            lnh_mpe_pr = lnh_results['mpe/precision'].round(3)
            lnh_mpe_rc = lnh_results['mpe/recall'].round(3)
            lnh_mpe_f1 = lnh_results['mpe/f1-score'].round(3)
            # Add subtitle
            ax[2, 1].axis('on')
            ax[2, 1].set_xticks([])
            ax[2, 1].get_yaxis().set_visible(False)
            ax[2, 1].set_xlabel(f'Lin. H.Avg. - P: {lnh_mpe_pr} | R: {lnh_mpe_rc} | F1: {lnh_mpe_f1}')

            # Add a global colorbar
            #fig.colorbar(ax[0, 1].get_images()[0], ax=ax.ravel().tolist())

            # Minimize free space
            fig.tight_layout()

            # Open the figure manually
            plt.show(block=False)

            print('Press ENTER to continue...')
            # Wait for keyboard input
            while plt.waitforbuttonpress() != True:
                continue

            # Close figure
            plt.close(fig)

    # Print a header for average results across all tracks of the dataset
    print(f'\tAverage Results ({eval_set.name()}):')

    # Print average results
    print(f'\t\t-(lin-cqt-1): {ln1_evaluator.average_results()[0]}')
    print(f'\t\t-(log-cqt-1): {lg1_evaluator.average_results()[0]}')
    print(f'\t\t-(lin-cqt-h): {lnh_evaluator.average_results()[0]}')
    print(f'\t\t-(log-cqt-h): {lgh_evaluator.average_results()[0]}')
    print(f'\t\t-(ss-mpe): {ss_evaluator.average_results()[0]}')