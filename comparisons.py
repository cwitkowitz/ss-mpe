# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from common import ComboSet
from FreeMusicArchive import FreeMusicArchive
from MagnaTagATune import MagnaTagATune
from NSynth import NSynth
from ToyNSynth import ToyNSynthTrain, ToyNSynthEval
from Bach10 import Bach10
from Su import Su
from TRIOS import TRIOS
from MusicNet import MusicNet
from model import SAUNet
from lhvqt import LHVQT
from objectives import *
from utils import *
from evaluate import MultipitchEvaluator

# Regular imports
from torch.utils.tensorboard import SummaryWriter
from sacred.observers import FileStorageObserver
from torch.utils.data import DataLoader
from torch_audiomentations import *
from sacred import Experiment
from tqdm import tqdm

import librosa
import torch
import os


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

from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH

import tensorflow as tf

# Load the BasicPitch model checkpoint corresponding to paper
basic_pitch = tf.saved_model.load(str(ICASSP_2022_MODEL_PATH))

# Load a checkpoint of the SS-MPE model
ss_mpe = torch.load(os.path.join('.', 'generated', 'experiments', 'Debugging', 'models', 'model-5.pt'), map_location='cpu')

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

# Loop through all evaluation datasets
for test_set in [bach10, su, trios, musicnet]:
    # Initialize evaluators for all models
    bp_evaluator = MultipitchEvaluator()
    ss_evaluator = MultipitchEvaluator()
    ln_evaluator = MultipitchEvaluator()
    sc_evaluator = MultipitchEvaluator()

    print(f'Evaluating {test_set.name()}...')

    # Loop through all tracks in the test set
    for i, track in enumerate(tqdm(test_set.tracks)):
        # Obtain a path for the track's audio
        audio_path = test_set.get_audio_path(track)
        # Extract the audio and ground-truth
        audio, ground_truth = test_set[i]

        # Obtain predictions from the BasicPitch model
        model_output, _, _ = predict(audio_path, basic_pitch)
        bp_salience = model_output

        pass
