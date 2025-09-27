# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from timbre_trap.datasets.MixedMultiPitch import URMP as URMP_Mixtures
from ss_mpe.datasets.AudioMixtures import E_GMD

from ss_mpe.objectives import *
from ss_mpe.framework import SS_MPE
from timbre_trap.utils import *

# Regular imports
from scipy.signal import convolve2d
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import torch
import os


# Name of the model to evaluate
ex_name = 'URMP_SPV_T_G_P_LR5E-4_2_BS8_MC3_W100_TTFC'

# Choose the model checkpoint to compare
checkpoint = 9000

# Choose the GPU on which to perform evaluation
gpu_id = None

# File layout of system (0 - desktop | 1 - lab)
path_layout = 0

# Construct the path to the top-level directory of the experiment
if path_layout == 1:
    experiment_dir = os.path.join('/', 'storage', 'frank', 'self-supervised-pitch', ex_name)
else:
    experiment_dir = os.path.join('..', '..', 'generated', 'experiments', ex_name)

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

# Set the URMP validation set in accordance with the MT3 paper
urmp_val_splits = ['01', '02', '12', '13', '24', '25', '31', '38', '39']

# Allocate remaining tracks to URMP training set
urmp_train_splits = URMP_Mixtures.available_splits()

for t in urmp_val_splits:
    # Remove validation tracks
    urmp_train_splits.remove(t)

# Instantiate URMP dataset mixtures for training
urmp_mixes_train = URMP_Mixtures(base_dir=None,
                                 splits=urmp_train_splits,
                                 sample_rate=sample_rate,
                                 cqt=ss_mpe.hcqt,
                                 n_secs=4,
                                 seed=seed)

# Sample from which to start NSynth analysis
urmp_start_idx = 20 # higher polyphony

# Slice NSynth dataset
urmp_mixes_train.tracks = urmp_mixes_train.tracks[urmp_start_idx:]

""""""
# Instantiate E-GMD audio for percussion-invariance
egmd = E_GMD(base_dir=None,
             splits=['train'],
             sample_rate=sample_rate,
             n_secs=4,
             seed=seed)
""""""


###################
## Visualization ##
###################

# Create a directory for saving visualized harmonics
vis_dir = os.path.join('..', '..', 'generated', 'visualization', 'harmonics')

# Loop through all tracks in the test set
for i, data in enumerate(tqdm(urmp_mixes_train)):
    # Determine which track is being processed
    track = data[constants.KEY_TRACK]
    # Extract audio and add a batch dimension
    audio = data[constants.KEY_AUDIO].unsqueeze(0)

    # Compute full set of spectral features
    features = ss_mpe.get_all_features(audio)

    # Extract relevant feature sets
    features_db   = features['db'].squeeze(0)
    features_db_1 = features['db_1'].squeeze(0)
    features_db_h = features['db_h'].squeeze(0)

    if i == 0:
        # Seed everything with the same seed
        seed_everything(0)

    # Determine training sequence length in frames
    n_frames = int(4 * sample_rate / ss_mpe.hcqt.hop_length)

    # Define maximum time and frequency shift
    max_shift_v = 2 * ss_mpe.hcqt.bins_per_octave
    max_shift_h = n_frames // 4

    # Maximum rate by which audio can be sped up or slowed down
    max_stretch_factor = 2

    """
    # Sample a track of percussion audio
    percussion_audio = egmd.get_audio(egmd.tracks[torch.randint(len(egmd), (1,))])
    # Superimpose percussion audio onto original audio
    percussion_audio = audio + egmd.slice_audio(percussion_audio.to(device), audio.size(-1))[0].unsqueeze(0)
    # Compute spectral features for percussion audio mixture
    features_db = ss_mpe.hcqt.to_decibels(ss_mpe.hcqt(percussion_audio)).squeeze(0)
    """

    #_, (vs, hs, sfs) = apply_random_transformations(features_db.unsqueeze(0), max_shift_v, max_shift_h, max_stretch_factor)
    """"""
    features_db, (vs, hs, sfs) = apply_random_transformations(features_db.unsqueeze(0), max_shift_v, max_shift_h, max_stretch_factor)
    features_db = features_db.squeeze(0)
    """"""

    # Transcribe the audio using the SS-MPE model
    #ss_activations = to_array(ss_mpe.transcribe(audio).squeeze())
    ss_activations = to_array(ss_mpe(features_db.unsqueeze(0))[0]).squeeze(0)

    # Extract ground-truth pitch salience activations
    gt_activations = data[constants.KEY_GROUND_TRUTH]

    # Widen the activations for easier visualization
    #gt_activations = convolve2d(gt_activations,
    #                            np.array([[0.5, 1, 0.5]]).T, 'same')

    # Apply vertical and horizontal translations, inserting zero at empties
    """"""
    gt_activations = to_array(apply_geometric_transformations(torch.Tensor(gt_activations).unsqueeze(0).unsqueeze(0), vs, hs, sfs).squeeze(0).squeeze(0))
    """"""
    ss_activations_t_ev = to_array(apply_geometric_transformations(torch.Tensor(ss_activations).unsqueeze(0).unsqueeze(0), vs, hs, sfs).squeeze(0).squeeze(0))

    # Create a directory for saving visualized harmonics
    save_dir = os.path.join(vis_dir, track)
    # Create the root directory
    os.makedirs(save_dir, exist_ok=True)

    # Loop through harmonic CQT features
    for (h, cqt) in zip(ss_mpe.hcqt.harmonics, features_db):
        # Open a new figure
        fig = initialize_figure(figsize=(4, 3))
        # Plot spectral features
        fig = plot_magnitude(cqt, fig=fig)
        # Minimize free space
        fig.tight_layout()
        # Construct path under visualization directory
        #save_path = os.path.join(save_dir, f'{track}_h{h}.jpg')
        #save_path = os.path.join(save_dir, f'{track}_h{h}-P.jpg')
        save_path = os.path.join(save_dir, f'{track}_h{h}-G.jpg')
        # Save the figure with minimal whitespace
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0)

    # Loop through custom features, output, and ground-truth
    for (h, cqt) in zip(['h1\'', 'wavg', 'out', 'gt', 't_ev'],
                        [features_db_1, features_db_h,
                         ss_activations, gt_activations, ss_activations_t_ev]):
        # Open a new figure
        fig = initialize_figure(figsize=(4, 3))
        # Plot spectral features
        fig = plot_magnitude(cqt, fig=fig)
        # Minimize free space
        fig.tight_layout()
        # Construct path under visualization directory
        #save_path = os.path.join(save_dir, f'{track}_{h}.jpg')
        #save_path = os.path.join(save_dir, f'{track}_{h}-P.jpg')
        save_path = os.path.join(save_dir, f'{track}_{h}-G.jpg')
        # Save the figure with minimal whitespace
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0)

    """
    # Open the figure manually
    plt.show(block=False)

    # Wait for keyboard input
    while plt.waitforbuttonpress() != True:
        continue
    """

    # Close figure
    plt.close('all')
