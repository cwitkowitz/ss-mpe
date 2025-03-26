# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from timbre_trap.datasets.MixedMultiPitch import Bach10, URMP, Su, TRIOS, MusicNet
from timbre_trap.datasets.SoloMultiPitch import GuitarSet
from timbre_trap.datasets.NoteDataset import NoteDataset
from ss_mpe.datasets.SoloMultiPitch import NSynth
from timbre_trap.framework import TimbreTrap

from ss_mpe.framework import HCQT
from timbre_trap.utils import *

# Regular imports
from tqdm import tqdm

import numpy as np
import mir_eval
import librosa
import torch
import os


# Disable CUDA to prevent cryptic errors
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Choose the GPU on which to perform evaluation
#gpu_id = None
gpu_id = 0

# Initialize a device pointer for loading the models
device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() and gpu_id is not None else 'cpu')

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
from basic_pitch.inference import predict, Model
from basic_pitch import ICASSP_2022_MODEL_PATH

# Number of bins in a single octave
bp_bins_per_octave = 36
# Load the Basic-Pitch model checkpoint corresponding to paper
basic_pitch_model = Model(ICASSP_2022_MODEL_PATH)
# Compute the MIDI frequency associated with each bin of Basic-Pitch predictions
bp_midi_freqs = librosa.note_to_midi('A0') + np.arange(264) / (bp_bins_per_octave / 12)


# Specify the names of the files to download from GitHub
script_name, weights_name = 'predict_on_audio.py', 'multif0.h5'
# Construct a path to a top-level directory for DeepSalience
deep_salience_dir = os.path.join('..', 'generated', 'deep_salience')
# Create the necessary file hierarchy if it doesn't exist
os.makedirs(os.path.join(deep_salience_dir, 'weights'), exist_ok=True)
# Construct paths for the downloaded files
script_path = os.path.join(deep_salience_dir, script_name)
weights_path = os.path.join(deep_salience_dir, 'weights', weights_name)
try:
    # Attempt to import the DeepSalience inference code
    from generated.deep_salience.predict_on_audio import (model_def,
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
    from generated.deep_salience.predict_on_audio import (model_def,
                                                          compute_hcqt,
                                                          get_single_test_prediction,
                                                          get_multif0)
# Initialize Deep-Salience
deep_salience = model_def()
# Load the weights from the paper
deep_salience.load_weights(weights_path)

# Construct a path to a top-level directory for ComparingDeepModelsMPE
deep_models_dir = os.path.join('..', 'generated', 'deep_models')
# Specify the names of the files to download from GitHub
model_script_name = 'unet_cnns.py'
hcqt_script_name = 'hcqt.py'
data_script_name = 'hcqt_datasets.py'
expr_name  = 'RETRAIN4_exp195f_musicnet_aligned_unet_extremelylarge_polyphony_softmax_rerun1.pt'
# Create the necessary file hierarchy if it doesn't exist
os.makedirs(os.path.join(deep_models_dir, 'models_pretrained'), exist_ok=True)
# Construct paths for the downloaded files
model_script_path = os.path.join(deep_models_dir, model_script_name)
hcqt_script_path = os.path.join(deep_models_dir, hcqt_script_name)
data_script_path = os.path.join(deep_models_dir, data_script_name)
weights_path = os.path.join(deep_models_dir, 'models_pretrained', expr_name)
try:
    # Attempt to import the model and preprocessing code from ComparingDeepModelsMPE
    from generated.deep_models.unet_cnns import simple_u_net_polyphony_classif_softmax
    from generated.deep_models.hcqt import compute_efficient_hcqt
    from generated.deep_models.hcqt_datasets import dataset_context
except ModuleNotFoundError:
    # Point to the top-level directory containing files to download
    url_dir = 'https://raw.githubusercontent.com/christofw/multipitch_architectures/master'
    # Construct the URLs of the files to download
    model_script_url = f'{url_dir}/libdl/nn_models/{model_script_name}'
    hcqt_script_url = f'{url_dir}/libdl/data_preprocessing/{hcqt_script_name}'
    data_script_url = f'{url_dir}/libdl/data_loaders/{data_script_name}'
    weights_url = f'{url_dir}/models_pretrained/{expr_name}'
    # Download the script and weights files
    stream_url_resource(model_script_url, model_script_path)
    stream_url_resource(hcqt_script_url, hcqt_script_path)
    stream_url_resource(data_script_url, data_script_path)
    stream_url_resource(weights_url, weights_path)

    # Fix one of the import lines
    with open(hcqt_script_path, 'r+') as f:
        # Read all the code lines
        lines = f.readlines()  # Get a list of all lines
        # Reset file pointer
        f.seek(0)
        # Update lines of code
        lines[1] = 'import numpy as np\n'
        lines[121] = '    tuning_est = librosa.estimate_tuning(y=f_audio, bins_per_octave=bins_per_octave)\n'
        # Stop processing file
        f.truncate()
        # Overwrite the code
        f.writelines(lines)

    # Fix one of the import lines
    with open(data_script_path, 'r+') as f:
        # Read all the code lines
        lines = f.readlines()  # Get a list of all lines
        # Reset file pointer
        f.seek(0)
        # Remove unnecessary imports
        lines.pop(5)
        # Stop processing file
        f.truncate()
        # Overwrite the code
        f.writelines(lines)

    # Retry the original imports
    from generated.deep_models.unet_cnns import simple_u_net_polyphony_classif_softmax
    from generated.deep_models.hcqt import compute_efficient_hcqt
    from generated.deep_models.hcqt_datasets import dataset_context

# Initialize polyphony U-Net architecture
PUnet_XL = simple_u_net_polyphony_classif_softmax(n_chan_input=6,
                                                  n_chan_layers=[128, 180, 150, 100],
                                                  #n_ch_out=2,
                                                  n_bins_in=6 * 12 * 3,
                                                  n_bins_out=72,
                                                  a_lrelu=0.3,
                                                  p_dropout=0.2,
                                                  scalefac=2,
                                                  num_polyphony_steps=24).to(device)

# Define path to pre-trained weights for recommended MusicNet split (test set MuN-10full)
model_path = os.path.join(deep_models_dir, 'models_pretrained', expr_name)

# Load checkpoint, add to device, and switch to evaluation mode
PUnet_XL.load_state_dict(torch.load(model_path))
PUnet_XL.eval()


# Construct the path to the final model checkpoint for the base Timbre-Trap model
model_path = os.path.join('..', '..', 'timbre-trap', 'generated', 'experiments', 'Base', 'models', 'model-8750.pt')

# Initialize autoencoder model and train from scratch
tt_mpe = TimbreTrap(sample_rate=22050,
                    n_octaves=9,
                    bins_per_octave=60,
                    secs_per_block=3,
                    latent_size=128,
                    model_complexity=2,
                    skip_connections=False)

# Load final checkpoint of the base Timbre-Trap model
tt_mpe.load_state_dict(torch.load(model_path, map_location=device))
tt_mpe.to(device)
tt_mpe.eval()

# Frequencies associated with Timbre-Trap estimates
tt_midi_freqs = tt_mpe.sliCQ.get_midi_freqs()
# Determine which Timbre-Trap bins correspond to valid frequencies for mir_eval
tt_invalid_freqs = librosa.midi_to_hz(tt_midi_freqs) > mir_eval.multipitch.MAX_FREQ


"""
import crepe

# Determine cent values for each bin of CREPE predictions
# (https://github.com/marl/crepe/blob/master/crepe/core.py#L103)
cents = np.linspace(0, 7180, 360) + 1997.3794084376191
# Convert cents to frequencies in Hz
cr_hz_freqs = 10 * 2 ** (cents / 1200)
# Compute the MIDI frequency for each bin
cr_midi_freqs = librosa.hz_to_midi(cr_hz_freqs)


import pesto

# Compute the MIDI frequency associated with each bin of PESTO predictions
pe_midi_freqs = torch.arange(384) / 3
# Determine which Timbre-Trap bins correspond to valid frequencies for mir_eval
pe_invalid_freqs = librosa.midi_to_hz(pe_midi_freqs) < mir_eval.multipitch.MIN_FREQ
"""


##############
## DATASETS ##
##############

# Point to the datasets within the storage drive containing them or use the default location
urmp_base_dir   = os.path.join('/', 'storage', 'frank', 'URMP') if path_layout else None
nsynth_base_dir = os.path.join('/', 'storageNVME', 'frank', 'NSynth') if path_layout else None
bch10_base_dir  = os.path.join('/', 'storage', 'frank', 'Bach10') if path_layout else None
su_base_dir     = os.path.join('/', 'storage', 'frank', 'Su') if path_layout else None
trios_base_dir  = os.path.join('/', 'storage', 'frank', 'TRIOS') if path_layout else None
mnet_base_dir   = os.path.join('/', 'storageNVME', 'frank', 'MusicNet') if path_layout else None
gset_base_dir   = os.path.join('/', 'storage', 'frank', 'GuitarSet') if path_layout else None

# Instantiate NSynth validation split for validation
nsynth_val = NSynth(base_dir=nsynth_base_dir,
                    splits=['valid'],
                    n_tracks=200,
                    sample_rate=sample_rate,
                    cqt=hcqt)

# Instantiate Bach10 dataset mixtures for evaluation
bch10_test = Bach10(base_dir=bch10_base_dir,
                    splits=None,
                    sample_rate=sample_rate,
                    cqt=hcqt)

# Set the URMP validation set in accordance with the MT3 paper
urmp_val_splits = ['01', '02', '12', '13', '24', '25', '31', '38', '39']
# Instantiate URMP dataset mixtures for validation
urmp_val = URMP(base_dir=urmp_base_dir,
                splits=urmp_val_splits,
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

# Instantiate MusicNet dataset mixtures for evaluation
mnet_test = MusicNet(base_dir=mnet_base_dir,
                     splits=['test'],
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
for eval_set in [urmp_val, nsynth_val, bch10_test, su_test, trios_test, mnet_test, gset_test]:
    # Initialize evaluators for each algorithm/model
    bp_evaluator = MultipitchEvaluator()
    ds_evaluator = MultipitchEvaluator()
    pu_evaluator = MultipitchEvaluator()
    tt_evaluator = MultipitchEvaluator()
    #cr_evaluator = MultipitchEvaluator()
    #pe_evaluator = MultipitchEvaluator()

    print_and_log(f'Results for {eval_set.name()}:', save_path)

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

        """
        # Obtain predictions from the BasicPitch model
        model_output, _, _ = predict(audio_path, basic_pitch_model)
        # Extract the pitch salience predictions
        bp_salience = model_output['contour'].T
        # Determine times associated with each frame of predictions
        bp_times = model_frames_to_time(bp_salience.shape[-1])
        # Apply peak-picking and thresholding on the raw salience
        bp_salience = threshold(filter_non_peaks(bp_salience), 0.3)
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
        """

        # Load the audio using librosa for consistency
        audio_lib, fs_lib = librosa.load(audio_path, sr=22050)

        #n_bins_PUnet = 3 * 12 * 6
        #min_pitch_PUnet = 24

        f_hcqt, fs_hcqt, hopsize_cqt = compute_efficient_hcqt(audio_lib,
                                                              fs=22050,
                                                              fmin=librosa.note_to_hz('C1'),
                                                              fs_hcqt_target=50,
                                                              bins_per_octave=3 * 12,
                                                              num_octaves=6,
                                                              num_harmonics=5,
                                                              num_subharmonics=1,
                                                              center_bins=True)

        test_params = {'batch_size': 1,
                       'shuffle': False,
                       'num_workers': 1
                       }

        test_dataset_params = {'context': 75,
                               'stride': 1,
                               'compression': 10
                               }
        half_context = test_dataset_params['context'] // 2

        inputs = np.transpose(f_hcqt, (2, 1, 0))
        targets = np.zeros(inputs.shape[1:])  # need dummy targets to use dataset object

        inputs_context = torch.from_numpy(np.pad(inputs, ((0, 0), (half_context, half_context + 1), (0, 0))))
        targets_context = torch.from_numpy(np.pad(targets, ((half_context, half_context + 1), (0, 0))))

        test_set = dataset_context(inputs_context, targets_context, test_dataset_params)
        test_generator = torch.utils.data.DataLoader(test_set, **test_params)

        pred_tot = np.zeros((0, 72))

        #max_frames = 160
        #k = 0
        for test_batch, test_labels in tqdm(test_generator, position=0, leave=True):
            #k += 1
            #if k > max_frames:
            #    break
            # Model computations
            y_pred, n_pred = PUnet_XL(test_batch.to(device))
            pred_log = torch.squeeze(torch.squeeze(y_pred, 2), 1).cpu().detach().numpy()
            # pred_log = torch.squeeze(y_pred.to('cpu')).detach().numpy()
            pred_tot = np.append(pred_tot, pred_log, axis=0)


        pu_midi_freqs = 24 + np.arange(72)
        pu_times = np.arange(pred_tot.shape[0]) / fs_hcqt

        pu_salience = pred_tot.T
        # Apply peak-picking and thresholding on the raw salience
        #pu_salience = threshold(filter_non_peaks(pu_salience), 0.4)
        # Apply thresholding on the raw salience
        pu_salience = threshold(pu_salience, 0.4)
        # Convert the activations to frame-level multi-pitch estimates
        pu_multi_pitch = eval_set.activations_to_multi_pitch(pu_salience, pu_midi_freqs)
        # Compute results for PUnet:XL predictions
        pu_results = pu_evaluator.evaluate(pu_times, pu_multi_pitch, times_ref, multi_pitch_ref)
        # Store results for this track
        pu_evaluator.append_results(pu_results)

        if verbose:
            # Print results for the individual track
            print_and_log(f'\t\t-(pu-netx): {pu_results}', save_path)


        # Extract audio and add to the appropriate device
        audio = data[constants.KEY_AUDIO].to(device).unsqueeze(0)

        """
        # Pad audio to next multiple of block length
        audio_padded = tt_mpe.sliCQ.pad_to_block_length(audio)
        # Determine the times associated with features
        times_est = tt_mpe.sliCQ.get_times(tt_mpe.sliCQ.get_expected_frames(audio_padded.size(-1)))
        # Transcribe the audio using the Timbre-Trap model
        tt_salience = to_array(tt_mpe.to_activations(tt_mpe.inference(audio_padded, True)).squeeze())
        # Peak-pick and threshold the Timbre-Trap activations
        tt_salience = threshold(filter_non_peaks(tt_salience), 0.5)
        # Remove activations for invalid frequencies
        tt_salience[tt_invalid_freqs] = 0
        # Convert the Timbre-Trap activations to frame-level multi-pitch estimates
        tt_multi_pitch = eval_set.activations_to_multi_pitch(tt_salience, tt_midi_freqs)
        # Compute results for predictions from the Timbre-Trap methodology
        tt_results = tt_evaluator.evaluate(times_est, tt_multi_pitch, times_ref, multi_pitch_ref)
        # Store results for this track
        tt_evaluator.append_results(tt_results)

        if verbose:
            # Print results for the individual track
            print_and_log(f'\t\t-(tt-mpe): {tt_results}', save_path)
        """

        """
        # Obtain salience predictions from the CREPE model
        cr_times, _, _, cr_salience = crepe.predict(to_array(audio.squeeze()), sample_rate, viterbi=False)
        # Apply peak-picking and thresholding on the raw salience
        cr_salience = threshold(filter_non_peaks(cr_salience.T), 0.3)
        # Convert the activations to frame-level multi-pitch estimates
        cr_multi_pitch = eval_set.activations_to_multi_pitch(cr_salience, cr_midi_freqs)
        # Compute results for BasicPitch predictions
        cr_results = cr_evaluator.evaluate(cr_times, cr_multi_pitch, times_ref, multi_pitch_ref)
        # Store results for this track
        cr_evaluator.append_results(cr_results)

        if verbose:
            # Print results for the individual track
            print_and_log(f'\t\t-(crepe): {cr_results}', save_path)


        # Obtain salience predictions from the PESTO model
        pe_times, pe_preds, _, pe_salience = pesto.predict(audio.squeeze(), sample_rate)
        # Convert time steps to seconds
        pe_times = to_array(pe_times) / 1000
        # Apply peak-picking and thresholding on the raw salience
        pe_salience = threshold(filter_non_peaks(pe_salience.T), 0.3)
        # Remove activations for invalid frequencies
        pe_salience[pe_invalid_freqs] = 0
        # Convert the activations to frame-level multi-pitch estimates
        pe_multi_pitch = eval_set.activations_to_multi_pitch(pe_salience, pe_midi_freqs)
        # Compute results for BasicPitch predictions
        pe_results = pe_evaluator.evaluate(pe_times, pe_multi_pitch, times_ref, multi_pitch_ref)
        # Store results for this track
        pe_evaluator.append_results(pe_results)

        if verbose:
            # Print results for the individual track
            print_and_log(f'\t\t-(pesto): {pe_results}', save_path)
        """

    # Print a header for average results across all tracks of the dataset
    print_and_log(f'\tAverage Results ({eval_set.name()}):', save_path)

    # Print average results
    print_and_log(f'\t\t-(bsc-ptc): {bp_evaluator.average_results()[0]}', save_path)
    print_and_log(f'\t\t-(dp-slnc): {ds_evaluator.average_results()[0]}', save_path)
    print_and_log(f'\t\t-(tt-mpe): {tt_evaluator.average_results()[0]}', save_path)
    #print_and_log(f'\t\t-(crepe): {cr_evaluator.average_results()[0]}', save_path)
    #print_and_log(f'\t\t-(pesto): {pe_evaluator.average_results()[0]}', save_path)
