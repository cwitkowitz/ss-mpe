# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from timbre_trap.datasets.MixedMultiPitch import URMP as URMP_Mixtures
from ss_mpe.datasets.AudioMixtures import E_GMD

from ss_mpe.framework.objectives import apply_translation, apply_distortion
from ss_mpe.framework import SS_MPE
from timbre_trap.utils import *

# Regular imports
from tqdm import tqdm

import matplotlib.pyplot as plt
import torch
import os


# Name of the model to evaluate
ex_name = 'URMP_SU_0'

# Choose the model checkpoint to compare
checkpoint = 1000

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
urmp_start_idx = 20

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

    """
    if i == 1:
        # Seed everything with the same seed
        seed_everything(0)

    # Determine training sequence length in frames
    n_frames = int(4 * sample_rate / ss_mpe.hcqt.hop_length)

    # Define maximum time and frequency shift
    max_shift_v = 2 * ss_mpe.hcqt.bins_per_octave
    max_shift_h = n_frames // 4

    # Maximum rate by which audio can be sped up or slowed down
    max_stretch_factor = 2

    # Sample a random vertical / horizontal shift for each sample in the batch
    v_shifts = torch.randint(low=-max_shift_v, high=max_shift_v + 1, size=(1,))-30
    h_shifts = torch.randint(low=-max_shift_h, high=max_shift_h + 1, size=(1,))

    # Compute inverse of maximum stretch factor
    min_stretch_factor = 1 / max_stretch_factor

    # Sample a random stretch factor for each sample in the batch, starting at minimum
    stretch_factors, stretch_factors_ = min_stretch_factor, torch.rand(size=(1,))
    # Split sampled values into piecewise ranges
    neg_perc = 2 * stretch_factors_.clip(max=0.5)
    pos_perc = 2 * (stretch_factors_ - 0.5).relu()
    # Scale stretch factor evenly across effective range
    stretch_factors += neg_perc * (1 - min_stretch_factor)
    stretch_factors += pos_perc * (max_stretch_factor - 1)

    # Apply vertical and horizontal translations, inserting zero at empties
    features_db = apply_translation(features_db.unsqueeze(0), v_shifts, axis=-2, val=0)
    features_db = apply_translation(features_db, h_shifts, axis=-1, val=0)
    # Apply time distortion, maintaining original dimensionality and padding with zeros
    features_db = apply_distortion(features_db, stretch_factors)

    # Process transformed features with provided model
    ss_activations = to_array(ss_mpe(features_db)[0].squeeze())

    features_db = features_db.squeeze(0)

    # Apply vertical and horizontal translations, inserting zero at empties
    transformed_output = apply_translation(ss_mpe.transcribe(audio), v_shifts, axis=-2, val=0)
    transformed_output = apply_translation(transformed_output, h_shifts, axis=-1, val=0)
    # Apply time distortion, maintaining original dimensionality and padding with zeros
    transformed_output = apply_distortion(transformed_output.unsqueeze(0), stretch_factors).squeeze().cpu().detach().numpy()
    """

    """"""
    # Sample a track of percussion audio
    percussion_audio = egmd.get_audio(egmd.tracks[torch.randint(len(egmd), (1,))])
    # Superimpose percussion audio onto original audio
    percussion_audio = audio + egmd.slice_audio(percussion_audio.to(device), audio.size(-1))[0].unsqueeze(0)
    # Compute spectral features for percussion audio mixture
    features_db = ss_mpe.hcqt.to_decibels(ss_mpe.hcqt(percussion_audio)).squeeze(0)
    """"""

    """"""
    # Transcribe the audio using the SS-MPE model
    ss_activations = to_array(ss_mpe.transcribe(audio).squeeze())
    """"""

    # Extract ground-truth pitch salience activations
    gt_activations = data[constants.KEY_GROUND_TRUTH]

    """
    # Apply vertical and horizontal translations, inserting zero at empties
    gt_activations = apply_translation(torch.Tensor(gt_activations).unsqueeze(0), v_shifts, axis=-2, val=0)
    gt_activations = apply_translation(gt_activations, h_shifts, axis=-1, val=0)
    # Apply time distortion, maintaining original dimensionality and padding with zeros
    gt_activations = apply_distortion(gt_activations.unsqueeze(0), stretch_factors).squeeze().cpu().detach().numpy()
    """

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
        save_path = os.path.join(save_dir, f'{track}_h{h}.pdf')
        # Save the figure with minimal whitespace
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0)

    # Loop through custom features, output, and ground-truth
    for (h, cqt) in zip(['h1\'', 'wavg', 'out', 'pgt', 'equiv'],
                        [features_db_1, features_db_h,
                         ss_activations, gt_activations]):#, transformed_output]):
        # Open a new figure
        fig = initialize_figure(figsize=(4, 3))
        # Plot spectral features
        fig = plot_magnitude(cqt, fig=fig)
        # Minimize free space
        fig.tight_layout()
        # Construct path under visualization directory
        save_path = os.path.join(save_dir, f'{track}_{h}.pdf')
        # Save the figure with minimal whitespace
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0)

    # Open the figure manually
    plt.show(block=False)

    # Wait for keyboard input
    while plt.waitforbuttonpress() != True:
        continue

    # Close figure
    plt.close('all')
