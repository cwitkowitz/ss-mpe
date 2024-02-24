# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from timbre_trap.datasets.MixedMultiPitch import Bach10, URMP, Su, TRIOS
from timbre_trap.datasets.SoloMultiPitch import GuitarSet
from timbre_trap.datasets.NoteDataset import NoteDataset
from ss_mpe.datasets.SoloMultiPitch import NSynth

from ss_mpe.framework import SS_MPE
from timbre_trap.utils import *

# Regular imports
from tqdm import tqdm

import librosa
import torch
import os


# Name of the model to evaluate
ex_name = '<EXPERIMENT_DIR>'
#ex_name = 'SS-MPE'
#ex_name = 'Timbre'
#ex_name = 'Geometric'
#ex_name = 'Energy'

# Choose the model checkpoint to compare
checkpoint = 0
#checkpoint = 37000
#checkpoint = 41750
#checkpoint = 37000
#checkpoint = 43000

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


##############
## DATASETS ##
##############

# Point to the datasets within the storage drive containing them or use the default location
nsynth_base_dir    = os.path.join('/', 'storageNVME', 'frank', 'NSynth') if path_layout else None
bch10_base_dir     = os.path.join('/', 'storage', 'frank', 'Bach10') if path_layout else None
urmp_base_dir      = os.path.join('/', 'storage', 'frank', 'URMP') if path_layout else None
su_base_dir        = os.path.join('/', 'storage', 'frank', 'Su') if path_layout else None
trios_base_dir     = os.path.join('/', 'storage', 'frank', 'TRIOS') if path_layout else None
gset_base_dir      = os.path.join('/', 'storage', 'frank', 'GuitarSet') if path_layout else None

# Instantiate NSynth validation split for validation
nsynth_val = NSynth(base_dir=nsynth_base_dir,
                    splits=['valid'],
                    n_tracks=200,
                    sample_rate=sample_rate,
                    cqt=ss_mpe.hcqt,
                    seed=seed)

# Instantiate Bach10 dataset mixtures for evaluation
bch10_test = Bach10(base_dir=bch10_base_dir,
                    splits=None,
                    sample_rate=sample_rate,
                    cqt=ss_mpe.hcqt)

# Instantiate URMP dataset mixtures for evaluation
urmp_test = URMP(base_dir=urmp_base_dir,
                 splits=None,
                 sample_rate=sample_rate,
                 cqt=ss_mpe.hcqt)

# Instantiate Su dataset for evaluation
su_test = Su(base_dir=su_base_dir,
             splits=None,
             sample_rate=sample_rate,
             cqt=ss_mpe.hcqt)

# Instantiate TRIOS dataset for evaluation
trios_test = TRIOS(base_dir=trios_base_dir,
                   splits=None,
                   sample_rate=sample_rate,
                   cqt=ss_mpe.hcqt)

# Instantiate GuitarSet dataset for evaluation
gset_test = GuitarSet(base_dir=gset_base_dir,
                      splits=None,
                      sample_rate=sample_rate,
                      cqt=ss_mpe.hcqt)


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
for eval_set in [nsynth_val, bch10_test, urmp_test, su_test, trios_test, gset_test]:
    # Initialize evaluators for each algorithm/model
    ln1_evaluator = MultipitchEvaluator()
    lg1_evaluator = MultipitchEvaluator()
    lnh_evaluator = MultipitchEvaluator()
    lgh_evaluator = MultipitchEvaluator()
    ss_evaluator = MultipitchEvaluator()

    print_and_log(f'Results for {eval_set.name()}:', save_path)

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
            print_and_log(f'\tResults for track \'{track}\' ({eval_set.name()}):', save_path)

        # Determine the times associated with predictions
        times_est = eval_set.cqt.get_times(eval_set.cqt.get_expected_frames(audio.size(-1)))

        # Compute full set of spectral features
        features = ss_mpe.get_all_features(audio)

        # Extract relevant feature sets
        features_pw_1 = features['pw_1']
        features_db_1 = features['db_1']
        features_pw_h = features['pw_h']
        features_db_h = features['db_h']

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
            print_and_log(f'\t\t-(lin-cqt-1): {ln1_results}', save_path)


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
            print_and_log(f'\t\t-(log-cqt-1): {lg1_results}', save_path)

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
            print_and_log(f'\t\t-(lin-cqt-h): {lnh_results}', save_path)


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


    # Print a header for average results across all tracks of the dataset
    print_and_log(f'\tAverage Results ({eval_set.name()}):', save_path)

    # Print average results
    print_and_log(f'\t\t-(lin-cqt-1): {ln1_evaluator.average_results()[0]}', save_path)
    print_and_log(f'\t\t-(log-cqt-1): {lg1_evaluator.average_results()[0]}', save_path)
    print_and_log(f'\t\t-(lin-cqt-h): {lnh_evaluator.average_results()[0]}', save_path)
    print_and_log(f'\t\t-(log-cqt-h): {lgh_evaluator.average_results()[0]}', save_path)
    print_and_log(f'\t\t-(ss-mpe): {ss_evaluator.average_results()[0]}', save_path)
