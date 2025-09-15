# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from timbre_trap.datasets.MixedMultiPitch import Bach10, URMP, Su, TRIOS, MusicNet
from timbre_trap.datasets.SoloMultiPitch import GuitarSet

from ss_mpe.framework import SS_MPE
from timbre_trap.utils import *

# Regular imports
from scipy.signal import convolve2d
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import torch
import os


# Experiment and model checkpoint to use as reference
best_ex_name, checkpoint = 'base/URMP_SPV_T_G_P_LR5E-4_2_BS8_MC3_W100_TTFC', 9000

# Two-stage experiment to use to illustrate degeneration
two_stage_ex_name = 'two-stage/URMP_SPV_T_G_P_-_+FMA_LR1E-4_2_BS24_R0.66_MC3_W100_TTFC'

# Choose the GPU on which to perform evaluation
gpu_id = None

# Flag to print results for each track separately
verbose = True

# File layout of system (0 - desktop | 1 - lab)
path_layout = 0

# Construct the path to the top-level directory of the experiment
if path_layout == 1:
    experiments_dir = os.path.join('/', 'storage', 'frank', 'self-supervised-pitch')
else:
    #experiments_dir = os.path.join('..', 'generated', 'experiments')
    experiments_dir = f'/media/rockstar/Icarus/ss-mpe_ISMIR_overfitting_degeneration'

# Set randomization seed
seed = 0

# Seed everything with the same seed
seed_everything(seed)

plt.rcParams.update({'axes.titlesize': 'x-large'})


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
model_path = os.path.join(experiments_dir, best_ex_name, 'models', f'model-{checkpoint}.pt')

# Load a checkpoint of the SS-MPE model
ss_mpe = SS_MPE.load(model_path, device=device)
ss_mpe.eval()

# Initialize a dictionary to hold the models
models = dict()

for cp in [2500, 5000, 7500]:#, 10000]:
    # Construct the path to the model checkpoint to evaluate
    model_path = os.path.join(experiments_dir, two_stage_ex_name, 'models', f'model-{cp}.pt')

    # Load the model at the specified checkpoint
    model = SS_MPE.load(model_path, device=device)
    model.eval()

    # Add the loaded model to dictionary
    models.update({f'+FMA-16-FT@{cp}' : model})


########################
## FEATURE EXTRACTION ##
########################

# Number of samples per second of audio
sample_rate = 22050

# Get HCQT of last model (should all be the same)
hcqt = model.hcqt

# Extract necessary parameters
fmin = hcqt.get_midi_freqs()[0]
fmax = fmin + hcqt.n_bins / (hcqt.bins_per_octave / 12)


#############
## DATASET ##
#############

# Point to the datasets within the storage drive containing them or use the default location
urmp_base_dir   = os.path.join('/', 'storage', 'frank', 'URMP') if path_layout else None
nsynth_base_dir = os.path.join('/', 'storageNVME', 'frank', 'NSynth') if path_layout else None
bch10_base_dir  = os.path.join('/', 'storage', 'frank', 'Bach10') if path_layout else None
su_base_dir     = os.path.join('/', 'storage', 'frank', 'Su') if path_layout else None
trios_base_dir  = os.path.join('/', 'storage', 'frank', 'TRIOS') if path_layout else None
mnet_base_dir   = os.path.join('/', 'storageNVME', 'frank', 'MusicNet') if path_layout else None
gset_base_dir   = os.path.join('/', 'storage', 'frank', 'GuitarSet') if path_layout else None

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

# Instantiate TRIOS dataset for evaluation
trios_test = TRIOS(base_dir=trios_base_dir,
                   splits=None,
                   sample_rate=sample_rate,
                   cqt=ss_mpe.hcqt)

# Instantiate MusicNet dataset mixtures for evaluation
mnet_test = MusicNet(base_dir=mnet_base_dir,
                     splits=['test'],
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

# Create a directory for saving inference visualization
save_dir = os.path.join('..', '..', 'generated', 'visualization', 'degeneration')

# Create the visualization directory
os.makedirs(save_dir, exist_ok=True)

# Loop through inference datasets
for eval_set in [bch10_test, su_test, trios_test, mnet_test, gset_test]:
    # Loop through all tracks in the test set
    for i, data in enumerate(tqdm(eval_set)):
        # Determine which track is being processed
        track = data[constants.KEY_TRACK]
        # Extract audio and add to the appropriate device
        audio = data[constants.KEY_AUDIO].to(device).unsqueeze(0)

        # Compute full set of spectral features
        features = model.get_all_features(audio)

        # Extract first harmonic of input HCQT
        features_db_1 = features['db_1'].squeeze(0)
        features_db_h = features['db_h'].squeeze(0)

        # Extract ground-truth pitch salience activations
        gt_activations = data[constants.KEY_GROUND_TRUTH]

        # Widen the activations for improved visualization
        gt_activations = convolve2d(gt_activations,
                                    np.array([[0.125, 1, 0.125]]).T, 'same')

        # Initialize a new figure with subplots
        fig, axes = plt.subplots(1, 5, figsize=(18, 4))

        # Determine track's attributes
        #name, pitch, vel = track.split('-')
        # Add a global title above all sub-plots
        #fig.suptitle(f'Track: {name} | Pitch: {pitch} | Velocity: {vel}')

        # Define extent for magnitude plots
        extent_midi = [0, audio.size(-1) / sample_rate, fmin, fmax]

        # Plot ground-truth activations
        fig.sca(axes[0])
        plot_magnitude(gt_activations, extent=extent_midi, fig=fig)
        # Axis management
        axes[0].set_title('Ground-Truth')
        #axes[0].axis('on')
        axes[0].axis('off')
        #axes[0].set_xlabel('')
        #axes[0].set_ylabel('')
        #axes[0].set_ylabel('Frequency (MIDI)')
        #axes[0].set_yticks([30, 40, 50, 60, 70, 80, 90, 100])
        #ax[0].grid()

        ss_activations = to_array(ss_mpe.transcribe(audio).squeeze(0))
        fig.sca(axes[1])
        plot_magnitude(ss_activations, extent=extent_midi, fig=fig)
        # Axis management
        #axes[1].axis('on')
        axes[1].axis('off')
        axes[1].set_title('Ref.')
        #axes[1].set_ylabel('')
        #axes[1].set_yticks([30, 40, 50, 60, 70, 80, 90, 100])
        #axes[1].set_yticklabels(['' for t in axes[1].get_yticks()])

        for i, (tag, model) in enumerate(models.items()):
            ss_activations = to_array(model.transcribe(audio).squeeze(0))
            fig.sca(axes[i + 2])
            plot_magnitude(ss_activations, extent=extent_midi, fig=fig)
            # Axis management
            #axes[i + 2].axis('on')
            axes[i + 2].axis('off')
            axes[i + 2].set_title(tag)
            #axes[i + 2].set_ylabel('')
            #axes[i + 2].set_yticks([30, 40, 50, 60, 70, 80, 90, 100])
            #axes[i + 2].set_yticklabels(['' for t in axes[i + 2].get_yticks()])

        # Minimize free space
        fig.tight_layout()

        # Open the figure manually
        plt.show(block=False)

        # Wait for keyboard input
        while plt.waitforbuttonpress() != True:
            continue

        # Prompt user to save figure
        save = input('Save figure? (y/n)')

        if save == 'y':
            # Construct path under visualization directory
            save_path = os.path.join(save_dir, f'{track}.pdf')
            # Save the figure with minimal whitespace
            fig.savefig(save_path, bbox_inches='tight', pad_inches=0)

        # Close figure
        plt.close(fig)
