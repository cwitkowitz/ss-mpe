# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from timbre_trap.datasets.NoteDataset import NoteDataset
from ss_mpe.datasets.SoloMultiPitch import NSynth

from ss_mpe.models import SS_MPE
from timbre_trap.utils import *

# Regular imports
from tqdm import tqdm

import matplotlib.pyplot as plt
import librosa
import torch
import os


# Name of the model to evaluate
ex_name = 'SS-MPE'

# Choose the model checkpoint to compare
checkpoint = 37000

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


###########
## MODEL ##
###########

# Initialize a device pointer for loading the models
device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() and gpu_id is not None else 'cpu')

# Construct the path to the model checkpoint to evaluate
model_path = os.path.join(experiment_dir, 'models', f'model-{checkpoint}.pt')

# Load a checkpoint of the SS-MPE model
ss_mpe = SS_MPE.load(model_path, device=device)
ss_mpe.eval()


#############
## DATASET ##
#############

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

# Weak/missing fundamental
#nsynth_start_idx = 21 # Synthetic Bass 34 30 50
#nsynth_start_idx = 99 # Electronic Keyboard 1 31 75
#nsynth_start_idx = 162 # Electronic Organ 57 25 25
#nsynth_start_idx = 192 # Acoustic String 80 28 127

# Strong sub-harmonic
#nsynth_start_idx = 60 # Synthetic Flute 0 64 50
#nsynth_start_idx = 147 # Electronic Organ 1 79 25

# Note played again sharply
#nsynth_start_idx = 35 # Synthetic Bass 134 34 25

# Aduible release of note
#nsynth_start_idx = 104 # Electronic Keyboard 1 86 25

# Weak attack
#nsynth_start_idx = 161 # Electronic Organ 28 82 50

# Gradual decay
#nsynth_start_idx = 142 # Acoustic Mallet 62 33 75
#nsynth_start_idx = 187 # Acoustic String 14 44 75

# Electronic effects
#nsynth_start_idx = 117 # Electronic Keyboard 3 63 100

# Strong artifacts
#nsynth_start_idx = 34 # Synthetic Bass 98 99 50

# Fluctuation outside nominal pitch
#nsynth_start_idx = 72 # Acoustic Guitar 15 38 100
#nsynth_start_idx = 91 # Electronic Guitar 28 21 25
#nsynth_start_idx = 180 # Acoustic Reed 23 42 100

# Overpowering artifacts
#nsynth_start_idx = 38 # Synthetic Bass 134 86 25
#nsynth_start_idx = 106 # Electronic Keyboard 1 106 75

# Simultaneous pitches
#nsynth_start_idx = 199 # Synthetic Vocal 3 104 100

# Corrupted samples
#nsynth_start_idx = 58 # Acoustic Brass 46 89 100

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
        features_pw_1 = features['pw_1']
        features_db_1 = features['db_1']
        features_pw_h = features['pw_h']
        features_db_h = features['db_h']

        # Extract ground-truth pitch salience activations
        gt_activations = data[constants.KEY_GROUND_TRUTH]

        # Peak-pick and threshold the linear-scaled magnitude
        ln1_activations = threshold(filter_non_peaks(to_array(features_pw_1)), 0.5)
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
        lg1_activations = threshold(filter_non_peaks(to_array(features_db_1)), 0.9)
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
        lnh_activations = threshold(filter_non_peaks(to_array(features_pw_h)), 0.3)
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
        lgh_activations = threshold(filter_non_peaks(to_array(features_db_h)), 0.9)
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
            # Initialize a new figure with subplots
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
            ss_mpe_pr = round(ss_results['mpe/precision'], 3)
            ss_mpe_rc = round(ss_results['mpe/recall'], 3)
            ss_mpe_f1 = round(ss_results['mpe/f1-score'], 3)
            # Add subtitle
            ax[0, 0].axis('on')
            ax[0, 0].set_xticks([])
            ax[0, 0].get_yaxis().set_visible(False)
            ax[0, 0].set_xlabel(f'SS-MPE - P: {ss_mpe_pr} | R: {ss_mpe_rc} | F1: {ss_mpe_f1}')

            # Plot 1st harmonic (log) activations
            fig.sca(ax[1, 0])
            plot_magnitude(features_db_1[0], fig=fig)
            # Extract 1st harmonic (log) performance measures
            lg1_mpe_pr = round(lg1_results['mpe/precision'], 3)
            lg1_mpe_rc = round(lg1_results['mpe/recall'], 3)
            lg1_mpe_f1 = round(lg1_results['mpe/f1-score'], 3)
            # Add subtitle
            ax[1, 0].axis('on')
            ax[1, 0].set_xticks([])
            ax[1, 0].get_yaxis().set_visible(False)
            ax[1, 0].set_xlabel(f'Log 1st - P: {lg1_mpe_pr} | R: {lg1_mpe_rc} | F1: {lg1_mpe_f1}')

            # Plot 1st harmonic (linear) activations
            fig.sca(ax[1, 1])
            plot_magnitude(features_pw_1[0], fig=fig)
            # Extract 1st harmonic (linear) performance measures
            ln1_mpe_pr = round(ln1_results['mpe/precision'], 3)
            ln1_mpe_rc = round(ln1_results['mpe/recall'], 3)
            ln1_mpe_f1 = round(ln1_results['mpe/f1-score'], 3)
            # Add subtitle
            ax[1, 1].axis('on')
            ax[1, 1].set_xticks([])
            ax[1, 1].get_yaxis().set_visible(False)
            ax[1, 1].set_xlabel(f'Lin. 1st - P: {ln1_mpe_pr} | R: {ln1_mpe_rc} | F1: {ln1_mpe_f1}')

            # Plot harmonic average (log) activations
            fig.sca(ax[2, 0])
            plot_magnitude(features_db_h[0], fig=fig)
            # Extract harmonic average (log) performance measures
            lgh_mpe_pr = round(lgh_results['mpe/precision'], 3)
            lgh_mpe_rc = round(lgh_results['mpe/recall'], 3)
            lgh_mpe_f1 = round(lgh_results['mpe/f1-score'], 3)
            # Add subtitle
            ax[2, 0].axis('on')
            ax[2, 0].set_xticks([])
            ax[2, 0].get_yaxis().set_visible(False)
            ax[2, 0].set_xlabel(f'Log H.Avg. - P: {lgh_mpe_pr} | R: {lgh_mpe_rc} | F1: {lgh_mpe_f1}')

            # Plot harmonic average (linear) activations
            fig.sca(ax[2, 1])
            plot_magnitude(features_pw_h[0], fig=fig)
            # Extract harmonic average (linear) performance measures
            lnh_mpe_pr = round(lnh_results['mpe/precision'], 3)
            lnh_mpe_rc = round(lnh_results['mpe/recall'], 3)
            lnh_mpe_f1 = round(lnh_results['mpe/f1-score'], 3)
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

            print('Press SPACE with plot focused to continue...')
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
