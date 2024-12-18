# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from timbre_trap.datasets.MixedMultiPitch import (URMP,
                                                  Bach10,
                                                  MusicNet,
                                                  Su,
                                                  TRIOS)
from timbre_trap.datasets.SoloMultiPitch import (MAESTRO,
                                                 GuitarSet)
from timbre_trap.datasets.AudioMixtures import (FMA,
                                                MedleyDB)
from timbre_trap.datasets import ComboDataset

from ss_mpe.datasets.SoloMultiPitch import NSynth
from ss_mpe.datasets.AudioMixtures import E_GMD

from ss_mpe.framework import SS_MPE, TT_Enc
from ss_mpe.objectives import *
from timbre_trap.utils import *
from evaluate import evaluate
from utils import *

# Regular imports
from torch.utils.tensorboard import SummaryWriter
from sacred.observers import FileStorageObserver
from torch.utils.data import DataLoader
from sacred import Experiment
from random import shuffle
from tqdm import tqdm

import numpy as np
import warnings
import librosa
import torch
import math
import os


CONFIG = 0 # (0 - desktop | 1 - lab)
EX_NAME = '_'.join(['FMA_EG_T_G_SPR'])

ex = Experiment('Train a model to perform MPE with self-supervised objectives only')


@ex.config
def config():
    ##############################
    ## TRAINING HYPERPARAMETERS ##
    ##############################

    # Specify a checkpoint from which to resume training (None to disable)
    checkpoint_path = None

    # Maximum number of training iterations to conduct
    max_epochs = 5000

    # Number of iterations between checkpoints
    checkpoint_interval = 250

    # Number of samples to gather for a batch
    batch_size = 20

    # Number of seconds of audio per sample
    n_secs = 4

    # Initial learning rate
    learning_rate = 5e-4

    # Scaling factors for each loss term
    multipliers = {
        'energy' : 1,
        'support' : 0,
        'harmonic' : 0,
        'sparsity' : 1,
        'entropy' : 0,
        'timbre' : 1,
        'geometric' : 1,
        'percussion' : 0,
        'noise' : 0,
        'feature' : 0,
        'supervised' : 0
    }

    # Perform augmentations on input features for energy-based and/or supervised objectives
    augment_features = False

    # Number of epochs spanning warmup phase (0 to disable)
    n_epochs_warmup = 0

    # Set validation dataset to compare for learning rate decay and early stopping
    validation_criteria_set = URMP.name()

    # Set validation metric to compare for learning rate decay and early stopping
    validation_criteria_metric = 'mpe/f1-score'

    # Select whether the validation criteria should be maximized or minimized
    validation_criteria_maximize = True # (False - minimize | True - maximize)

    # Number of epochs without improvement before reducing learning rate (0 to disable)
    n_epochs_decay = 0

    # Number of epochs before starting epoch counter for learning rate decay
    n_epochs_cooldown = 0

    # Number of epochs without improvement before early stopping (None to disable)
    n_epochs_early_stop = None

    # IDs of the GPUs to use, if available
    gpu_ids = [0]

    # Random seed for this experiment
    seed = 0

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

    # Harmonics to stack along channel dimension
    harmonics = [0.5, 1, 2, 3, 4, 5]

    ############
    ## OTHERS ##
    ############

    # Number of threads to use for data loading
    n_workers = 4 * len(gpu_ids)

    # Top-level directory under which to save all experiment files
    if CONFIG == 1:
        root_dir = os.path.join('/', 'storage', 'frank', 'ss-mpe_journal', EX_NAME)
    else:
        root_dir = os.path.join('..', 'generated', 'experiments', EX_NAME)

    # Create the root directory
    os.makedirs(root_dir, exist_ok=True)

    # Add a file storage observer for the log directory
    ex.observers.append(FileStorageObserver(root_dir))


@ex.automain
def train_model(checkpoint_path, max_epochs, checkpoint_interval, batch_size, n_secs, learning_rate, multipliers,
                augment_features, n_epochs_warmup, validation_criteria_set, validation_criteria_metric,
                validation_criteria_maximize, n_epochs_decay, n_epochs_cooldown, n_epochs_early_stop, gpu_ids,
                seed, sample_rate, hop_length, fmin, bins_per_octave, n_bins, harmonics, n_workers, root_dir):
    # Discard read-only types
    multipliers = dict(multipliers)
    harmonics = list(harmonics)
    gpu_ids = list(gpu_ids)

    # Seed everything with the same seed
    seed_everything(seed)

    # Initialize the primary PyTorch device
    device = torch.device(f'cuda:{gpu_ids[0]}'
                          if torch.cuda.is_available() else 'cpu')

    ########################
    ## FEATURE EXTRACTION ##
    ########################

    # Create weighting for harmonics (harmonic loss)
    harmonic_weights = 1 / torch.Tensor(harmonics) ** 2
    # Apply zero weight to sub-harmonics (harmonic loss)
    harmonic_weights[harmonic_weights > 1] = 0
    # Normalize the harmonic weights
    harmonic_weights /= torch.sum(harmonic_weights)
    # Add frequency and time dimensions for broadcasting
    harmonic_weights = harmonic_weights.unsqueeze(-1).unsqueeze(-1)
    # Make sure weights are on appropriate device
    harmonic_weights = harmonic_weights.to(device)

    # Pack together HCQT parameters for readability
    hcqt_params = {'sample_rate': sample_rate,
                   'hop_length': hop_length,
                   'fmin': fmin,
                   'bins_per_octave': bins_per_octave,
                   'n_bins': n_bins,
                   'gamma': None,
                   'harmonics': harmonics,
                   'weights' : harmonic_weights}

    # Infer the number of bins per semitone
    bins_per_semitone = bins_per_octave / 12

    # Determine maximum supported MIDI frequency
    fmax = fmin + n_bins / bins_per_semitone

    ###########
    ## MODEL ##
    ###########

    if checkpoint_path is None:
        # Initialize Timbre-Trap encoder
        model = TT_Enc(hcqt_params,
                       n_blocks=4,
                       model_complexity=2)
    else:
        # Load weights of the specified model checkpoint
        model = SS_MPE.load(checkpoint_path, device=device)

        for k, v in model.hcqt_params.items():
            # Get HCQT parameter
            p = hcqt_params[k]
            # Check if type is PyTorch Tensor
            isTensor = type(p) is torch.Tensor
            # Check for mismatch between parameters
            if not (v.equal(p) if isTensor else v == p):
                # Warn user of mismatch between specified/loaded parameters
                warnings.warn(f'Selected value for \'{k}\' does not '
                              'match saved value...', RuntimeWarning)

    if len(gpu_ids) > 1:
        # Wrap model for multi-GPU usage
        model = DataParallel(model, device_ids=gpu_ids)

    # Add model to primary device
    model = model.to(device)

    ##############
    ## DATASETS ##
    ##############

    # Audio dataset paths
    fma_base_dir    = os.path.join('/', 'storage', 'frank', 'FMA') if CONFIG else None
    mdb_base_dir    = os.path.join('/', 'storage', 'frank', 'MedleyDB') if CONFIG else None
    egmd_base_dir   = os.path.join('/', 'storageNVME', 'frank', 'E-GMD') if CONFIG else None
    nsynth_base_dir = os.path.join('/', 'storageNVME', 'frank', 'NSynth') if CONFIG else None

    # MPE dataset paths
    urmp_base_dir     = os.path.join('/', 'storageNVME', 'frank', 'URMP') if CONFIG else None
    bch10_base_dir    = os.path.join('/', 'storageNVME', 'frank', 'Bach10') if CONFIG else None
    gset_base_dir     = os.path.join('/', 'storageNVME', 'frank', 'GuitarSet') if CONFIG else None
    mdb_ptch_base_dir = os.path.join('/', 'storage', 'frank', 'MedleyDB-Pitch') if CONFIG else None

    # AMT dataset paths
    mstro_base_dir = os.path.join('/', 'storage', 'frank', 'MAESTRO') if CONFIG else None
    mnet_base_dir  = os.path.join('/', 'storageNVME', 'frank', 'MusicNet') if CONFIG else None
    su_base_dir    = os.path.join('/', 'storageNVME', 'frank', 'Su') if CONFIG else None
    trios_base_dir = os.path.join('/', 'storageNVME', 'frank', 'TRIOS') if CONFIG else None

    # Initialize lists to hold training datasets
    train_ss, train_sup, train_both = list(), list(), list()

    # Instantiate NSynth training split for training
    nsynth_train = NSynth(base_dir=nsynth_base_dir,
                          splits=['train'],
                          midi_range=np.array([fmin, fmax]),
                          sample_rate=sample_rate,
                          n_secs=n_secs,
                          seed=seed)
    #train_ss.append(nsynth_train)

    # Instantiate MusicNet audio (training) mixtures for training
    mnet_audio = MusicNet(base_dir=mnet_base_dir,
                          splits=['train'],
                          sample_rate=sample_rate,
                          n_secs=n_secs,
                          seed=seed)
    #train_ss.append(mnet_audio)

    # Use all available splits for FMA
    fma_genres_harmonic = FMA.available_splits()

    # Instantiate FMA audio mixtures for training
    fma_audio = FMA(base_dir=fma_base_dir,
                    splits=fma_genres_harmonic,
                    sample_rate=sample_rate,
                    n_secs=n_secs,
                    seed=seed)
    train_ss.append(fma_audio)

    # Instantiate MedleyDB audio mixtures for training
    """mdb_audio = MedleyDB(base_dir=mdb_base_dir,
                         sample_rate=sample_rate,
                         n_secs=n_secs,
                         seed=seed)"""
    #train_ss.append(mdb_audio)

    # Set the URMP validation set in accordance with the MT3 paper
    urmp_val_splits = ['01', '02', '12', '13', '24', '25', '31', '38', '39']

    # Allocate remaining tracks to URMP training set
    urmp_train_splits = URMP.available_splits()

    for t in urmp_val_splits:
        # Remove validation tracks
        urmp_train_splits.remove(t)


    # Instantiate URMP dataset mixtures for training
    urmp_mixes_train = URMP(base_dir=urmp_base_dir,
                            splits=urmp_train_splits,
                            sample_rate=sample_rate,
                            cqt=model.hcqt,
                            n_secs=n_secs,
                            seed=seed)
    #train_sup.append(urmp_mixes_train)
    #train_both.append(urmp_mixes_train)

    # Combine training datasets
    train_ss = ComboDataset(train_ss)
    train_sup = ComboDataset(train_sup)
    train_both = ComboDataset(train_both)

    # Ratio for self-supervised to supervised training data
    ss_ratio = 0.95

    # Default batch size and workers
    batch_size_ss, n_workers_ss = 0, 0
    batch_size_sup, n_workers_sup = 0, 0
    batch_size_both, n_workers_both = 0, 0

    if len(train_ss) and not len(train_sup) and not len(train_both):
        # All data is for self-supervised training
        batch_size_ss, n_workers_ss = batch_size, n_workers
    elif not len(train_ss) and len(train_sup) and not len(train_both):
        # All data is for supervised training
        batch_size_sup, n_workers_sup = batch_size, n_workers
    elif not len(train_ss) and not len(train_sup) and len(train_both):
        # All data is for both types of supervised training
        batch_size_both, n_workers_both = batch_size, n_workers
    elif len(train_ss) and len(train_sup) and not len(train_both):
        # Split data between self-supervised and supervised training
        batch_size_ss, n_workers_ss = round(ss_ratio * batch_size), round(ss_ratio * n_workers)
        batch_size_sup, n_workers_sup = round((1 - ss_ratio) * batch_size), round((1 - ss_ratio) * n_workers)
    elif len(train_ss) and not len(train_sup) and len(train_both):
        # Split data between self-supervised and both types of training
        batch_size_ss, n_workers_ss = round(ss_ratio * batch_size), round(ss_ratio * n_workers)
        batch_size_both, n_workers_both = round((1 - ss_ratio) * batch_size), round((1 - ss_ratio) * n_workers)
    elif not len(train_ss) and len(train_sup) and len(train_both):
        # Split data between supervised and both types of training
        batch_size_sup, n_workers_sup = round((1 - ss_ratio) * batch_size), round((1 - ss_ratio) * n_workers)
        batch_size_both, n_workers_both = round(ss_ratio * batch_size), round(ss_ratio * n_workers)
    else:
        # Split data between all combinations of training
        batch_size_ss, n_workers_ss = round(ss_ratio * batch_size), round(ss_ratio * n_workers)
        batch_size_sup, n_workers_sup = round(0.5 * (1 - ss_ratio) * batch_size), round(0.5 * (1 - ss_ratio) * n_workers)
        batch_size_both, n_workers_both = round(0.5 * (1 - ss_ratio) * batch_size), round(0.5 * (1 - ss_ratio) * n_workers)

    # Determine number of samples for each type of training
    n_ss = batch_size_ss + batch_size_both
    n_sup = batch_size_sup + batch_size_both

    # Default loaders to empty list
    loader_ss = list()
    loader_sup = list()
    loader_both = list()

    if batch_size_ss:
        # Initialize dataloader for self-supervised data
        loader_ss = DataLoader(dataset=train_ss,
                               batch_size=batch_size_ss,
                               shuffle=True,
                               num_workers=n_workers_ss,
                               pin_memory=True,
                               drop_last=True)

    if batch_size_sup:
        # Initialize dataloader for supervised data
        loader_sup = DataLoader(dataset=train_sup,
                                batch_size=batch_size_sup,
                                shuffle=True,
                                num_workers=n_workers_sup,
                                pin_memory=True,
                                drop_last=True)

    if batch_size_both:
        # Initialize dataloader for comprehensive data
        loader_both = DataLoader(dataset=train_both,
                                 batch_size=batch_size_both,
                                 shuffle=True,
                                 num_workers=n_workers_both,
                                 pin_memory=True,
                                 drop_last=True)

    if not len(loader_ss):
        # Add null entries for each batch of self-supervised training data
        loader_ss += [None] * max(len(loader_sup), len(loader_both))

    if not len(loader_sup):
        # Add null entries for each batch of supervised training data
        loader_sup += [None] * max(len(loader_ss), len(loader_both))

    if not len(loader_both):
        # Add null entries for each batch of both types training data
        loader_both += [None] * max(len(loader_ss), len(loader_sup))

    # Instantiate NSynth validation split for validation
    nsynth_val = NSynth(base_dir=nsynth_base_dir,
                        splits=['valid'],
                        n_tracks=200,
                        midi_range=np.array([fmin, fmax]),
                        sample_rate=sample_rate,
                        cqt=model.hcqt,
                        seed=seed)

    # Instantiate URMP dataset mixtures for validation
    urmp_val = URMP(base_dir=urmp_base_dir,
                    splits=urmp_val_splits,
                    sample_rate=sample_rate,
                    cqt=model.hcqt,
                    seed=seed)

    # Instantiate Bach10 dataset mixtures for evaluation
    bch10_test = Bach10(base_dir=bch10_base_dir,
                        splits=None,
                        sample_rate=sample_rate,
                        cqt=model.hcqt,
                        seed=seed)

    # Instantiate Su dataset for evaluation
    su_test = Su(base_dir=su_base_dir,
                 splits=None,
                 sample_rate=sample_rate,
                 cqt=model.hcqt,
                 seed=seed)

    # Instantiate TRIOS dataset for evaluation
    trios_test = TRIOS(base_dir=trios_base_dir,
                       splits=None,
                       sample_rate=sample_rate,
                       cqt=model.hcqt,
                       seed=seed)

    # Instantiate MusicNet dataset mixtures for evaluation
    mnet_test = MusicNet(base_dir=mnet_base_dir,
                         splits=['test'],
                         sample_rate=sample_rate,
                         cqt=model.hcqt,
                         seed=seed)

    # Instantiate GuitarSet dataset for validation
    gset_val = GuitarSet(base_dir=gset_base_dir,
                         splits=['05'],
                         sample_rate=sample_rate,
                         cqt=model.hcqt,
                         seed=seed)

    # Instantiate GuitarSet dataset for evaluation
    gset_test = GuitarSet(base_dir=gset_base_dir,
                          splits=None,
                          sample_rate=sample_rate,
                          cqt=model.hcqt,
                          seed=seed)

    # Instantiate MAESTRO dataset for validation
    mstro_val = MAESTRO(base_dir=mstro_base_dir,
                        splits=['validation'],
                        sample_rate=sample_rate,
                        n_secs=30,
                        cqt=model.hcqt,
                        seed=seed)
    shuffle(mstro_val.tracks)
    mstro_val.tracks = mstro_val.tracks[:10]

    # Instantiate MAESTRO dataset for evaluation
    mstro_test = MAESTRO(base_dir=mstro_base_dir,
                         splits=['test'],
                         sample_rate=sample_rate,
                         cqt=model.hcqt,
                         seed=seed)

    # Add all validation datasets to a list
    validation_sets = [urmp_val, nsynth_val, bch10_test, su_test, trios_test, mnet_test, gset_val, mstro_val]

    # Add all evaluation datasets to a list
    evaluation_sets = [bch10_test, su_test, trios_test, mnet_test, gset_test, mstro_test]

    #################
    ## PREPARATION ##
    #################

    # Initialize an optimizer for the model parameters with differential learning rates
    optimizer = torch.optim.AdamW([{'params' : model.encoder_parameters(), 'lr' : learning_rate}])
    #optimizer = torch.optim.SGD([{'params': model.encoder_parameters(), 'lr': learning_rate, 'momentum': 0.9}])

    # Determine the amount of batches in one epoch
    epoch_steps = min(len(loader_ss), len(loader_sup), len(loader_both))

    # Compute number of validation checkpoints corresponding to learning rate decay cooldown and window
    n_checkpoints_cooldown = math.ceil(n_epochs_cooldown * epoch_steps / checkpoint_interval)
    n_checkpoints_decay = math.ceil(n_epochs_decay * epoch_steps / checkpoint_interval)

    if n_epochs_early_stop is not None:
        # Compute number of validation checkpoints corresponding to early stopping window
        n_checkpoints_early_stop = math.ceil(n_epochs_early_stop * epoch_steps / checkpoint_interval)
    else:
        # Early stopping is disabled
        n_checkpoints_early_stop = None

    # Warmup global learning rate over a fixed number of steps according to a cosine function
    warmup_scheduler = CosineWarmup(optimizer, n_steps=n_epochs_warmup * epoch_steps)

    # Decay global learning rate by a factor of 1/2 after validation performance has plateaued
    decay_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                 mode='max' if validation_criteria_maximize else 'min',
                                                                 factor=0.5,
                                                                 patience=n_checkpoints_decay,
                                                                 threshold=1E-3,
                                                                 cooldown=n_checkpoints_cooldown)

    # Enable anomaly detection to debug any NaNs (can increase overhead)
    #torch.autograd.set_detect_anomaly(True)

    # Enable cuDNN auto-tuner to optimize CUDA kernel (might improve
    # performance, but adds initial overhead to find the best kernel)
    cudnn_benchmarking = False

    if cudnn_benchmarking:
        # Enable benchmarking prior to training
        torch.backends.cudnn.benchmark = True

    # Construct the path to the directory for saving models
    log_dir = os.path.join(root_dir, 'models')

    # Initialize a writer to log results
    writer = SummaryWriter(log_dir)

    # Number of batches that have been processed
    batch_count = 0

    # Keep track of the model with the best validation results
    best_model_checkpoint = None

    # Keep track of the best model's results for comparison
    best_results = None

    # Counter for number of checkpoints since previous best results
    n_checkpoints_elapsed = 0

    # Flag to indicate early stopping criteria has been met
    early_stop_criteria = False

    #################
    ## TIMBRE LOSS ##
    #################

    # Pointiness for parabolic equalization
    pointiness = 2

    # Set keyword arguments for parabolic equalization
    parabolic_kwargs = {
        'eq_fn' : sample_parabolic_equalization,
        'pointiness': pointiness
    }

    # Use random equalization
    eq_kwargs = parabolic_kwargs

    # Insert equalization density argument
    eq_kwargs.update({'density' : 1})

    ####################
    ## GEOMETRIC LOSS ##
    ####################

    # Determine training sequence length in frames
    n_frames = int(n_secs * sample_rate / hop_length)

    # Define maximum time and frequency shift
    max_shift_v = round(1 * bins_per_octave)
    max_shift_h = n_frames // 4

    # Maximum rate by which audio can be sped up or slowed down
    max_stretch_factor = 2

    # Set keyword arguments for geometric transformations
    gm_kwargs = {
        'max_shift_v' : max_shift_v,
        'max_shift_h' : max_shift_h,
        'max_stretch_factor' : max_stretch_factor
    }

    #####################
    ## PERCUSSION LOSS ##
    #####################

    # Initialize list to hold unpitched datasets
    percussive_sets = list()

    # Instantiate E-GMD audio for percussion-invariance
    egmd = E_GMD(base_dir=egmd_base_dir,
                 splits=['train'],
                 sample_rate=sample_rate,
                 seed=seed)
    percussive_sets.append(egmd)

    # Combine percussive and noise datasets
    percussive_set_combo = ComboDataset(percussive_sets)

    # Maximum volume of percussion relative to original audio
    max_volume = 1.0

    # Set keyword arguments for percussion mixtures
    pc_kwargs = {
        'percussive_set_combo' : percussive_set_combo,
        'max_volume' : max_volume
    }

    ################
    ## NOISE LOSS ##
    ################

    # Maximum volume of noise relative to original audio
    max_volume = 0.25

    # Set keyword arguments for additive noise
    an_kwargs = {
        'max_volume' : max_volume
    }

    ##################
    ## FEATURE LOSS ##
    ##################

    # Mode (0 - channel | 1 - frequency | 2 - bin) for dropout
    mode = 0

    # Set keyword arguments for dropout
    dp_kwargs = {
        'mode' : mode
    }

    ##############################
    ## TRAINING/VALIDATION LOOP ##
    ##############################

    # Loop through epochs
    for i in range(max_epochs):
        # Loop through batches of both types of data
        for (data_ss, data_sup, data_both) in tqdm(zip(loader_ss, loader_sup, loader_both), desc=f'Epoch {i + 1}'):
            # Increment the batch counter
            batch_count += 1

            # Log the current learning rates for this batch
            writer.add_scalar('train/loss/learning_rate/encoder', optimizer.param_groups[0]['lr'], batch_count)

            # Initialize a list for audio and ground-truth from all data partitions
            audio, ground_truth = list(), list()

            if data_ss is not None:
                # Extract self-supervised audio and add to appropriate device
                audio.append(data_ss[constants.KEY_AUDIO].to(device))

            if data_both is not None:
                # Extract both types data and add to appropriate device
                audio.append(data_both[constants.KEY_AUDIO].to(device))
                ground_truth.append(data_both[constants.KEY_GROUND_TRUTH].to(device))

            if data_sup is not None:
                # Extract supervised data and add to appropriate device
                audio.append(data_sup[constants.KEY_AUDIO].to(device))
                ground_truth.append(data_sup[constants.KEY_GROUND_TRUTH].to(device))

            # Combine audio from all data partitions
            audio = torch.cat(audio)
            # Combine audio from supervised data partitions
            ground_truth = torch.cat(ground_truth) if len(ground_truth) else None

            # Compute full set of spectral features
            features = model.get_all_features(audio)

            # Extract spectral features from original audio
            features_db = features['db']
            features_db_1 = features['db_1']
            features_db_h = features['db_h']

            if augment_features:
                # Superimpose percussion audio onto augmented audio
                audio_aug = mix_random_percussion(audio, **pc_kwargs)
                # Superimpose noise onto augmented audio
                audio_aug = add_random_noise(audio_aug, **an_kwargs)
                # Compute spectral features for augmented audio
                features_db_aug = model.get_all_features(audio_aug)['db']
                # Apply random equalizations to augmented audio
                features_db_aug, _ = apply_random_equalizations(features_db_aug, model.hcqt, **eq_kwargs)
                # Apply random geometric transformations to augmentation audio
                features_db_aug, (vs, hs, sfs) = apply_random_transformations(features_db_aug, **gm_kwargs)
                # Apply parallel geometric transformations to targets and ground-truth
                features_db_1 = apply_geometric_transformations(features_db_1.unsqueeze(1), vs, hs, sfs).squeeze(1)
                features_db_h = apply_geometric_transformations(features_db_h.unsqueeze(1), vs, hs, sfs).squeeze(1)
                ground_truth = apply_geometric_transformations(ground_truth.unsqueeze(1), vs, hs, sfs).squeeze(1)
                # Apply dropout to input features
                features_db_aug = drop_random_features(features_db_aug, **dp_kwargs)
                # Compute energy-based losses using augmented spectral features
                features_in = features_db_aug
            else:
                # Compute energy-based losses using original spectral features
                features_in = features_db

            with torch.autocast(device_type=f'cuda'):
                # Process features to obtain logits
                logits = model(features_in)
                # Convert to (implicit) pitch salience activations
                activations = torch.sigmoid(logits)

                # Compute energy loss w.r.t. weighted harmonic sum for the batch
                energy_loss = compute_energy_loss(logits[:n_ss], features_db_h[:n_ss]) if n_ss else torch.tensor(0.)
                # Log the energy loss for this batch
                writer.add_scalar('train/loss/energy', energy_loss.item(), batch_count)

                debug_nans(energy_loss, 'energy')

                # Compute support loss w.r.t. first harmonic for the batch
                support_loss = compute_support_loss(logits[:n_ss], features_db_1[:n_ss]) if n_ss else torch.tensor(0.)
                # Log the support loss for this batch
                writer.add_scalar('train/loss/support', support_loss.item(), batch_count)

                debug_nans(support_loss, 'support')

                # Compute harmonic loss w.r.t. weighted harmonic sum for the batch
                harmonic_loss = compute_harmonic_loss(logits[:n_ss], features_db_h[:n_ss]) if n_ss else torch.tensor(0.)
                # Log the harmonic loss for this batch
                writer.add_scalar('train/loss/harmonic', harmonic_loss.item(), batch_count)

                debug_nans(harmonic_loss, 'harmonic')

                # Compute sparsity loss for the batch
                sparsity_loss = compute_sparsity_loss(activations[:n_ss]) if n_ss else torch.tensor(0.)
                # Log the sparsity loss for this batch
                writer.add_scalar('train/loss/sparsity', sparsity_loss.item(), batch_count)

                debug_nans(sparsity_loss, 'sparsity')

                # Compute entropy loss for the batch
                entropy_loss = compute_entropy_loss(logits[:n_ss]) if n_ss else torch.tensor(0.)
                # Log the entropy loss for this batch
                writer.add_scalar('train/loss/entropy', entropy_loss.item(), batch_count)

                debug_nans(entropy_loss, 'entropy')

                # Compute timbre-invariance loss for the batch
                timbre_loss = compute_timbre_loss(model, features_db[:n_ss], activations[:n_ss], **eq_kwargs) if n_ss else torch.tensor(0.)
                # Log the timbre-invariance loss for this batch
                writer.add_scalar('train/loss/timbre', timbre_loss.item(), batch_count)

                debug_nans(timbre_loss, 'timbre')

                # Compute geometric-equivariance loss for the batch
                geometric_loss = compute_geometric_loss(model, features_db[:n_ss], activations[:n_ss], **gm_kwargs) if n_ss else torch.tensor(0.)
                # Log the geometric-equivariance loss for this batch
                writer.add_scalar('train/loss/geometric', geometric_loss.item(), batch_count)

                debug_nans(geometric_loss, 'geometric')

                # Compute percussion-invariance loss for the batch
                percussion_loss = compute_percussion_loss(model, audio[:n_ss], activations[:n_ss], **pc_kwargs) if n_ss else torch.tensor(0.)
                # Log the percussion-invariance loss for this batch
                writer.add_scalar('train/loss/percussion', percussion_loss.item(), batch_count)

                debug_nans(percussion_loss, 'percussion')

                # Compute noise-invariance loss for the batch
                noise_loss = compute_noise_loss(model, audio[:n_ss], activations[:n_ss], **an_kwargs) if n_ss else torch.tensor(0.)
                # Log the noise-invariance loss for this batch
                writer.add_scalar('train/loss/noise', noise_loss.item(), batch_count)

                debug_nans(noise_loss, 'noise')

                # Compute feature-invariance loss for the batch
                feature_loss = compute_feature_loss(model, features_db[:n_ss], activations[:n_ss]) if n_ss else torch.tensor(0.)
                # Log the feature-invariance loss for this batch
                # TODO - update to feature when convenient
                writer.add_scalar('train/loss/channel', feature_loss.item(), batch_count)

                debug_nans(feature_loss, 'feature')

                # Compute supervised BCE loss for the batch
                supervised_loss = compute_supervised_loss(logits[batch_size_ss:], ground_truth, False) if n_sup else torch.tensor(0.)
                # Log the supervised BCE loss for this batch
                writer.add_scalar('train/loss/supervised', supervised_loss.item(), batch_count)

                debug_nans(supervised_loss, 'supervised')

                # upward: (1 - cosine_anneal(batch_count, 1000 * epoch_steps, start=0, floor=0.))

                # Compute the total loss for this batch
                total_loss = multipliers['energy'] * energy_loss + \
                             multipliers['support'] * support_loss + \
                             multipliers['harmonic'] * harmonic_loss + \
                             multipliers['sparsity'] * sparsity_loss + \
                             multipliers['entropy'] * entropy_loss + \
                             multipliers['timbre'] * timbre_loss + \
                             multipliers['geometric'] * geometric_loss + \
                             multipliers['percussion'] * percussion_loss + \
                             multipliers['noise'] * noise_loss + \
                             multipliers['feature'] * feature_loss + \
                             multipliers['supervised'] * supervised_loss

                # Log the total loss for this batch
                writer.add_scalar('train/loss/total', total_loss.item(), batch_count)

                # Zero the accumulated gradients
                optimizer.zero_grad()
                # Compute gradients using total loss
                total_loss.backward()

                # Compute the average gradient norm across the encoder
                avg_norm_encoder = average_gradient_norms(model.encoder)
                # Log the average gradient norm of the encoder for this batch
                writer.add_scalar('train/avg_norm/encoder', avg_norm_encoder, batch_count)
                # Determine the maximum gradient norm across encoder
                max_norm_encoder = get_max_gradient_norm(model.encoder)
                # Log the maximum gradient norm of the encoder for this batch
                writer.add_scalar('train/max_norm/encoder', max_norm_encoder, batch_count)

                # Apply gradient clipping for training stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

                # Perform an optimization step
                optimizer.step()

            if warmup_scheduler.is_active():
                # Step the learning rate warmup scheduler
                warmup_scheduler.step()

            if batch_count % checkpoint_interval == 0:
                # Construct a path to save the model checkpoint
                model_path = os.path.join(log_dir, f'model-{batch_count}.pt')
                # Save model checkpoint
                model.save(model_path)

                if cudnn_benchmarking:
                    # Disable benchmarking prior to validation
                    torch.backends.cudnn.benchmark = False

                # Initialize dictionary to hold all validation results
                validation_results = dict()

                for val_set in validation_sets:
                    try:
                        # Validate the model checkpoint on each validation dataset
                        validation_results[val_set.name()] = evaluate(model=model,
                                                                      eval_set=val_set,
                                                                      multipliers=multipliers,
                                                                      writer=writer,
                                                                      i=batch_count,
                                                                      device=device,
                                                                      eq_kwargs=eq_kwargs,
                                                                      gm_kwargs=gm_kwargs,
                                                                      pc_kwargs=pc_kwargs,
                                                                      an_kwargs=an_kwargs,
                                                                      dp_kwargs=dp_kwargs)
                    except Exception as e:
                        print(f'Error validating \'{val_set.name()}\': {repr(e)}')

                # Make sure model is on correct device and switch to training mode
                model = model.to(device)
                model.train()

                if cudnn_benchmarking:
                    # Re-enable benchmarking after validation
                    torch.backends.cudnn.benchmark = True

                if decay_scheduler.patience and not warmup_scheduler.is_active():
                    # Step the learning rate decay scheduler by logging the validation metric for the checkpoint
                    decay_scheduler.step(validation_results[validation_criteria_set][validation_criteria_metric])

                # Extract the result on the specified metric from the validation results for comparison
                current_score = validation_results[validation_criteria_set][validation_criteria_metric]

                if best_results is not None:
                    # Extract the currently tracked best result on the specified metric for comparison
                    best_score = best_results[validation_criteria_set][validation_criteria_metric]

                if best_results is None or \
                        (validation_criteria_maximize and current_score > best_score) or \
                        (not validation_criteria_maximize and current_score < best_score):
                    print(f'New best at {batch_count} iterations...')

                    # Set current checkpoint as best
                    best_model_checkpoint = batch_count
                    # Update best results
                    best_results = validation_results
                    # Reset number of checkpoints
                    n_checkpoints_elapsed = 0
                else:
                    # Increment number of checkpoints
                    n_checkpoints_elapsed += 1

                if n_checkpoints_early_stop is not None and n_checkpoints_elapsed >= n_checkpoints_early_stop:
                    # Early stop criteria has been reached
                    early_stop_criteria = True

                    break

        if early_stop_criteria:
            # Stop training
            break

    print(f'Achieved best results at {best_model_checkpoint} iterations...')

    for val_set in validation_sets:
        # Log the results at the best checkpoint for each validation dataset in metrics.json
        ex.log_scalar(f'Validation Results ({val_set.name()})', best_results[val_set.name()], best_model_checkpoint)

    ################
    ## EVALUATION ##
    ################

    # Construct a path to the best model checkpoint
    best_model_path = os.path.join(log_dir, f'model-{best_model_checkpoint}.pt')
    # Load best model and make sure it is in evaluation mode
    best_model = SS_MPE.load(best_model_path, device=device)
    best_model.eval()

    if cudnn_benchmarking:
        # Disable benchmarking prior to evaluation
        torch.backends.cudnn.benchmark = False

    for eval_set in evaluation_sets:
        try:
            # Evaluate the model using testing split
            final_results = evaluate(model=best_model,
                                     eval_set=eval_set,
                                     multipliers=multipliers,
                                     #device=device,
                                     eq_kwargs=eq_kwargs,
                                     gm_kwargs=gm_kwargs,
                                     pc_kwargs=pc_kwargs,
                                     an_kwargs=an_kwargs,
                                     dp_kwargs=dp_kwargs)
        except Exception as e:
            print(f'Error evaluating \'{eval_set.name()}\': {repr(e)}')

        # Log the evaluation results for this dataset in metrics.json
        ex.log_scalar(f'Evaluation Results ({eval_set.name()})', final_results, best_model_checkpoint)
