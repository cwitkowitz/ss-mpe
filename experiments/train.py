# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from timbre_trap.datasets.MixedMultiPitch import URMP as URMP_Mixtures, Bach10 as Bach10_Mixtures, Su, TRIOS, MusicNet
from timbre_trap.datasets.SoloMultiPitch import GuitarSet
from timbre_trap.datasets.AudioMixtures import FMA, MedleyDB
from timbre_trap.datasets import ComboDataset, StemMixingDataset

from ss_mpe.datasets.SoloMultiPitch import NSynth
from ss_mpe.datasets.AudioMixtures import E_GMD

from ss_mpe.framework import SS_MPE, TT_Base
from ss_mpe.framework.objectives import *
from timbre_trap.utils import *
from evaluate import evaluate

# Regular imports
from torch.utils.tensorboard import SummaryWriter
from sacred.observers import FileStorageObserver
from torch.utils.data import DataLoader
from sacred import Experiment
from tqdm import tqdm

import numpy as np
import warnings
import librosa
import torch
import math
import os


DEBUG = 0 # (0 - off | 1 - on)
CONFIG = 0 # (0 - desktop | 1 - lab)
EX_NAME = '_'.join(['<EXPERIMENT_NAME>'])

ex = Experiment('Train a model to perform MPE with self-supervised objectives only')


@ex.config
def config():
    ##############################
    ## TRAINING HYPERPARAMETERS ##
    ##############################

    # Specify a checkpoint from which to resume training (None to disable)
    checkpoint_path = None

    # Maximum number of training iterations to conduct
    max_epochs = 10000

    # Number of iterations between checkpoints
    checkpoint_interval = 250

    # Number of samples to gather for a batch
    batch_size = 4 if DEBUG else 8

    # Number of seconds of audio per sample
    n_secs = 8

    # Initial learning rate for encoder
    learning_rate_encoder = 1e-4

    # Initial learning rate for decoder
    learning_rate_decoder = learning_rate_encoder

    # Group together both learning rates
    learning_rates = [learning_rate_encoder, learning_rate_decoder]

    # Scaling factors for each loss term
    multipliers = {
        'support' : 1,
        'harmonic' : 1,
        'sparsity' : 1,
        'timbre' : 1,
        'geometric' : 1,
        'unpitched' : 0,
        'supervised' : 1
    }

    # Number of epochs spanning warmup phase (0 to disable)
    n_epochs_warmup = 0

    # Set validation dataset to compare for learning rate decay and early stopping
    validation_criteria_set = NSynth.name()

    # Set validation metric to compare for learning rate decay and early stopping
    validation_criteria_metric = 'loss/total'

    # Select whether the validation criteria should be maximized or minimized
    validation_criteria_maximize = False # (False - minimize | True - maximize)

    # Late starting point (0 to disable)
    n_epochs_late_start = 0

    # Number of epochs without improvement before reducing learning rate (0 to disable)
    n_epochs_decay = 0.5

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

    # Harmonics to stack along channel dimension of HCQT
    harmonics = [0.5, 1, 2, 3, 4, 5]

    ############
    ## OTHERS ##
    ############

    # Number of threads to use for data loading
    n_workers = 0 if DEBUG else 8 * len(gpu_ids)

    # Top-level directory under which to save all experiment files
    if CONFIG == 1:
        root_dir = os.path.join('/', 'storage', 'frank', 'scaling_ss-mpe', EX_NAME)
    else:
        root_dir = os.path.join('..', 'generated', 'experiments', EX_NAME)

    # Create the root directory
    os.makedirs(root_dir, exist_ok=True)

    if DEBUG:
        # Print a warning message indicating debug mode is active
        warnings.warn('Running in DEBUG mode...', RuntimeWarning)

    # Add a file storage observer for the log directory
    ex.observers.append(FileStorageObserver(root_dir))


@ex.automain
def train_model(checkpoint_path, max_epochs, checkpoint_interval, batch_size, n_secs, learning_rates,
                multipliers, n_epochs_warmup, validation_criteria_set, validation_criteria_metric,
                validation_criteria_maximize, n_epochs_late_start, n_epochs_decay, n_epochs_cooldown,
                n_epochs_early_stop, gpu_ids, seed, sample_rate, hop_length, fmin, bins_per_octave,
                n_bins, harmonics, n_workers, root_dir):
    # Discard read-only types
    learning_rates = list(learning_rates)
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
        # Initialize autoencoder model
        model = TT_Base(hcqt_params,
                        latent_size=128,
                        model_complexity=2,
                        skip_connections=False)
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

    # Point to the datasets within the storage drive containing them or use the default location
    nsynth_base_dir = os.path.join('/', 'storageNVME', 'frank', 'NSynth') if CONFIG else None
    urmp_base_dir   = os.path.join('/', 'storage', 'frank', 'URMP') if CONFIG else None
    bch10_base_dir  = os.path.join('/', 'storage', 'frank', 'Bach10') if CONFIG else None
    su_base_dir     = os.path.join('/', 'storage', 'frank', 'Su') if CONFIG else None
    trios_base_dir  = os.path.join('/', 'storage', 'frank', 'TRIOS') if CONFIG else None
    gset_base_dir   = os.path.join('/', 'storage', 'frank', 'GuitarSet') if CONFIG else None
    fma_base_dir    = os.path.join('/', 'storageNVME', 'frank', 'FMA') if CONFIG else None
    mnet_base_dir   = os.path.join('/', 'storageNVME', 'frank', 'MusicNet') if CONFIG else None
    mydb_base_dir   = os.path.join('/', 'storage', 'frank', 'MedleyDB') if CONFIG else None
    egmd_base_dir   = os.path.join('/', 'storage', 'frank', 'E-GMD') if CONFIG else None

    # Initialize list to hold all training datasets
    all_train = list()

    # Set the URMP validation set in accordance with the MT3 paper
    urmp_val_splits = ['01', '02', '12', '13', '24', '25', '31', '38', '39']

    # Allocate remaining tracks to URMP training set
    urmp_train_splits = URMP_Mixtures.available_splits()

    for t in urmp_val_splits:
        # Remove validation tracks
        urmp_train_splits.remove(t)

    if DEBUG:
        # Instantiate NSynth validation split for training
        nsynth_stems_train = NSynth(base_dir=nsynth_base_dir,
                                    splits=['valid'],
                                    n_tracks=200,
                                    midi_range=np.array([fmin, fmax]),
                                    sample_rate=sample_rate,
                                    cqt=model.hcqt,
                                    n_secs=n_secs,
                                    seed=seed)
        all_train.append(nsynth_stems_train)
    else:
        # Instantiate NSynth training split for training
        nsynth_stems_train = NSynth(base_dir=nsynth_base_dir,
                                    splits=['train'],
                                    midi_range=np.array([fmin, fmax]),
                                    sample_rate=sample_rate,
                                    cqt=model.hcqt,
                                    n_secs=n_secs,
                                    seed=seed)
        #all_train.append(nsynth_stems_train)

        # Instantiate random NSynth stem mixtures for training
        nsynth_mixes_train = StemMixingDataset([nsynth_stems_train],
                                               tracks_per_epoch=100000,
                                               n_min=1,
                                               n_max=5,
                                               seed=seed)
        #all_train.append(nsynth_mixes_train)

        # Define mostly-harmonic splits for FMA
        fma_splits = ['Rock', 'Folk', 'Instrumental',
                      'Pop', 'Classical', 'Jazz',
                      'Country', 'Soul-RnB', 'Blues']

        # Instantiate FMA audio for training
        fma_train = FMA(base_dir=fma_base_dir,
                        splits=fma_splits,
                        sample_rate=sample_rate,
                        n_secs=n_secs,
                        seed=seed)
        #all_train.append(fma_train)

        # Instantiate URMP dataset mixtures for training
        urmp_mixes_train = URMP_Mixtures(base_dir=urmp_base_dir,
                                         splits=urmp_train_splits,
                                         sample_rate=sample_rate,
                                         cqt=model.hcqt,
                                         n_secs=n_secs,
                                         seed=seed)
        #for i in range(round(len(nsynth_stems_train) / len(urmp_mixes_train))):
        all_train.append(urmp_mixes_train)

        # Instantiate MusicNet audio for training
        mnet_train = MusicNet(base_dir=mnet_base_dir,
                              splits=None,
                              sample_rate=sample_rate,
                              n_secs=n_secs,
                              seed=seed)
        #all_train.append(mnet_train)

        # Instantiate MedleyDB audio for training
        mydb_train = MedleyDB(base_dir=mydb_base_dir,
                              splits=None,
                              sample_rate=sample_rate,
                              n_secs=n_secs,
                              seed=seed)
        #all_train.append(mydb_train)

    # Combine all training datasets
    all_train = ComboDataset(all_train)

    # Initialize a PyTorch dataloader
    loader = DataLoader(dataset=all_train,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=n_workers,
                        collate_fn=separate_ground_truth,
                        pin_memory=True,
                        drop_last=True)

    # Instantiate NSynth validation split for validation
    nsynth_val = NSynth(base_dir=nsynth_base_dir,
                        splits=['valid'],
                        n_tracks=200,
                        midi_range=np.array([fmin, fmax]),
                        sample_rate=sample_rate,
                        cqt=model.hcqt,
                        seed=seed)

    # Set the URMP validation set as was defined in the MT3 paper
    urmp_val_splits = ['01', '02', '12', '13', '24', '25', '31', '38', '39']
    # Instantiate URMP dataset mixtures for validation
    urmp_val = URMP_Mixtures(base_dir=urmp_base_dir,
                             splits=urmp_val_splits,
                             sample_rate=sample_rate,
                             cqt=model.hcqt,
                             seed=seed)

    # Instantiate GuitarSet dataset for validation
    gset_val = GuitarSet(base_dir=gset_base_dir,
                         splits=['05'],
                         sample_rate=sample_rate,
                         cqt=model.hcqt,
                         seed=seed)

    # Instantiate NSynth testing split for evaluation
    nsynth_test = NSynth(base_dir=nsynth_base_dir,
                         splits=['test'],
                         sample_rate=sample_rate,
                         cqt=model.hcqt,
                         seed=seed)

    # Instantiate Bach10 dataset mixtures for evaluation
    bch10_test = Bach10_Mixtures(base_dir=bch10_base_dir,
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

    # Instantiate URMP dataset mixtures for evaluation
    urmp_test = URMP_Mixtures(base_dir=urmp_base_dir,
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

    # Instantiate GuitarSet dataset for evaluation
    gset_test = GuitarSet(base_dir=gset_base_dir,
                          splits=None,
                          sample_rate=sample_rate,
                          cqt=model.hcqt,
                          seed=seed)

    # Instantiate MusicNet dataset for evaluation
    mnet_test = MusicNet(base_dir=mnet_base_dir,
                         splits=['test'],
                         sample_rate=sample_rate,
                         cqt=model.hcqt,
                         seed=seed)

    # Add all validation datasets to a list
    validation_sets = [nsynth_val, urmp_val, bch10_test, su_test, trios_test, gset_val, mnet_test]

    # Add all evaluation datasets to a list
    evaluation_sets = [nsynth_test, bch10_test, su_test, trios_test, urmp_test, gset_test, mnet_test]

    #################
    ## PREPARATION ##
    #################

    # Initialize an optimizer for the model parameters with differential learning rates
    optimizer = torch.optim.AdamW([{'params' : model.encoder_parameters(), 'lr' : learning_rates[0]},
                                   {'params' : model.decoder_parameters(), 'lr' : learning_rates[1]}])

    # Determine amount of batches in one epoch
    epoch_steps = len(loader)

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
                                                                 threshold=2E-3,
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

    # Number of random points to sample per octave
    points_per_octave = 2

    # Determine semitone span of frequency support
    semitone_span = model.hcqt.midi_freqs.max() - model.hcqt.midi_freqs.min()

    # Determine how many bins are represented across all harmonics
    n_psuedo_bins = (bins_per_semitone * semitone_span).round()

    # Determine how many octaves have been covered
    n_octaves = int(math.ceil(n_psuedo_bins / bins_per_octave))

    # Calculate number of cut/boost points to sample
    n_points = 1 + points_per_octave * n_octaves

    # Standard deviation of boost/cut
    std_dev = 0.10

    # Set keyword arguments for random equalization
    random_kwargs = {
        'n_points' : n_points,
        'std_dev' : std_dev
    }

    # Pointiness for parabolic equalization
    pointiness = 5

    # Set keyword arguments for parabolic equalization
    parabolic_kwargs = {
        'pointiness' : pointiness
    }

    # Maximum amplitude for Gaussian equalization
    max_A = 0.375

    # Maximum standard deviation for Gaussian equalization
    max_std_dev = 2 * bins_per_octave

    # Whether to sample fixed rather than varied shapes
    fixed_shape = False

    # Set keyword arguments for Gaussian equalization
    gaussian_kwargs = {
        'max_A' : max_A,
        'max_std_dev' : max_std_dev,
        'fixed_shape' : fixed_shape
    }

    # Set equalization type and corresponding parameter values
    eq_fn, eq_kwargs = sample_gaussian_equalization, gaussian_kwargs

    ####################
    ## GEOMETRIC LOSS ##
    ####################

    # Determine training sequence length in frames
    n_frames = int(n_secs * sample_rate / hop_length)

    # Define maximum time and frequency shift
    max_shift_v = 2 * bins_per_octave
    max_shift_h = n_frames // 4

    # Maximum rate by which audio can be sped up or slowed down
    max_stretch_factor = 2

    # Set keyword arguments for geometric transformations
    gm_kwargs = {
        'max_shift_v' : max_shift_v,
        'max_shift_h' : max_shift_h,
        'max_stretch_factor' : max_stretch_factor
    }

    ####################
    ## UNPITCHED LOSS ##
    ####################

    # Instantiate E-GMD audio for percussion-invariance
    egmd = E_GMD(base_dir=egmd_base_dir,
                 splits=['train'],
                 sample_rate=sample_rate,
                 n_secs=n_secs,
                 seed=seed)

    # TODO - combo set with other noise / percussion datasets

    # Initialize a PyTorch dataloader for unpitched data
    unpitched_loader = DataLoader(dataset=egmd,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=n_workers,
                                  pin_memory=True,
                                  drop_last=True)

    #####################
    ## SUPERVISED LOSS ##
    #####################

    # Compute the number of bins within a quarter tone
    bins_per_quarter = round(bins_per_semitone / 2)
    # Determine the number of bins for the blur kernel
    n_bins_blur = 1 + 2 * bins_per_quarter
    # Create the kernel for Gaussian blur
    blur = torch.signal.windows.gaussian(n_bins_blur, std=1., device=device)
    # Add appropriate dimensions and convert to double
    blur = blur.resize(1, 1, len(blur), 1).double()

    ##############################
    ## TRAINING/VALIDATION LOOP ##
    ##############################

    # Loop through epochs
    for i in range(max_epochs):
        # Loop through batches of audio
        for (data_mpe, data_audio, _) in tqdm(loader, desc=f'Epoch {i + 1}'):
            # Increment the batch counter
            batch_count += 1

            if warmup_scheduler.is_active():
                # Step the learning rate warmup scheduler
                warmup_scheduler.step()

            if data_mpe is not None:
                # Extract ground-truth pitch salience activations
                ground_truth = data_mpe[constants.KEY_GROUND_TRUTH].to(device)
                # Keep track of number of samples with ground-truth
                n_ground_truth = ground_truth.size(0)
                # Add a temporary channel dimension
                ground_truth = ground_truth.unsqueeze(-3)
                # Apply the Gaussian blurring kernel to the ground-truth salience
                ground_truth = torch.nn.functional.conv2d(ground_truth, blur, padding='same')
                # Remove the temporary channel dimension
                ground_truth = ground_truth.squeeze(-3)

            if data_mpe is None:
                # Extract audio samples from audio-only data
                audio = data_audio[constants.KEY_AUDIO]
            elif data_audio is None:
                # Extract audio samples from MPE data
                audio = data_mpe[constants.KEY_AUDIO]
            else:
                # Concatenate samples from both sets of data
                audio = torch.cat((data_mpe[constants.KEY_AUDIO],
                                   data_audio[constants.KEY_AUDIO]))

            # Add audio to appropriate device
            audio = audio.to(device)

            # Log the current learning rates for this batch
            writer.add_scalar('train/loss/learning_rate/encoder', optimizer.param_groups[0]['lr'], batch_count)
            writer.add_scalar('train/loss/learning_rate/decoder', optimizer.param_groups[1]['lr'], batch_count)

            # Compute full set of spectral features
            features = model.get_all_features(audio)

            # Extract relevant feature sets
            features_db   = features['db']
            features_db_1 = features['db_1']
            features_db_h = features['db_h']

            # Sample a batch of unpitched audio
            unpitched_data = next(iter(unpitched_loader))
            # Superimpose unpitched audio onto original audio
            unpitched_audio = audio + unpitched_data[constants.KEY_AUDIO].to(device)
            # Compute spectral features for unpitched audio mixture
            features_unp = model.hcqt.to_decibels(model.hcqt(unpitched_audio))

            with torch.autocast(device_type=f'cuda'):
                # Process features to obtain logits
                logits, _, losses = model(features_db)
                # Convert to (implicit) pitch salience activations
                estimate = torch.sigmoid(logits)

                # Compute support loss w.r.t. first harmonic for the batch
                support_loss = compute_support_loss(logits, features_db_1)
                # Log the support loss for this batch
                writer.add_scalar('train/loss/support', support_loss.item(), batch_count)

                debug_nans(support_loss, 'support')

                # Compute harmonic loss w.r.t. weighted harmonic sum for the batch
                harmonic_loss = compute_harmonic_loss(logits, features_db_h)
                # Log the harmonic loss for this batch
                writer.add_scalar('train/loss/harmonic', harmonic_loss.item(), batch_count)

                debug_nans(harmonic_loss, 'harmonic')

                # Compute sparsity loss for the batch
                sparsity_loss = compute_sparsity_loss(estimate)
                # Log the sparsity loss for this batch
                writer.add_scalar('train/loss/sparsity', sparsity_loss.item(), batch_count)

                debug_nans(sparsity_loss, 'sparsity')

                # Compute timbre-invariance loss for the batch
                timbre_loss = compute_timbre_loss(model, features_db, logits, eq_fn, **eq_kwargs)
                # Log the timbre-invariance loss for this batch
                writer.add_scalar('train/loss/timbre', timbre_loss.item(), batch_count)

                debug_nans(timbre_loss, 'timbre')

                # Compute geometric-invariance loss for the batch
                geometric_loss = compute_geometric_loss(model, features_db, logits, **gm_kwargs)
                # Log the geometric-invariance loss for this batch
                writer.add_scalar('train/loss/geometric', geometric_loss.item(), batch_count)

                debug_nans(geometric_loss, 'geometric')

                # Compute unpitched-invariance loss for the batch
                unpitched_loss = compute_unpitched_loss(model, features_unp, logits)
                # Log the unpitched-invariance loss for this batch
                writer.add_scalar('train/loss/unpitched', unpitched_loss.item(), batch_count)

                debug_nans(unpitched_loss, 'unpitched')

                # Compute the total loss for this batch
                total_loss = multipliers['support'] * support_loss + \
                             multipliers['harmonic'] * harmonic_loss + \
                             multipliers['sparsity'] * sparsity_loss + \
                             multipliers['timbre'] * timbre_loss + \
                             multipliers['geometric'] * geometric_loss + \
                             multipliers['unpitched'] * unpitched_loss

                if data_mpe is not None:
                    # Compute supervised BCE loss for the batch
                    supervised_loss = compute_supervised_loss(logits[:n_ground_truth], ground_truth, True)
                    # Log the supervised BCE loss for this batch
                    writer.add_scalar('train/loss/supervised', supervised_loss.item(), batch_count)
                    # Add supervised loss to the total loss
                    total_loss += multipliers['supervised'] * supervised_loss

                    debug_nans(supervised_loss, 'supervised')

                if i >= n_epochs_late_start:
                    # Currently no late-state losses
                    # TODO - remove if unnecessary
                    pass

                for key_loss, val_loss in losses.items():
                    # Log the model loss for this batch
                    writer.add_scalar(f'train/loss/{key_loss}', val_loss.item(), batch_count)
                    # Add the model loss to the total loss
                    total_loss += multipliers.get(key_loss, 1) * val_loss

                    debug_nans(val_loss, key_loss)

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

                # Compute the average gradient norm across the decoder
                avg_norm_decoder = average_gradient_norms(model.decoder)
                # Log the average gradient norm of the decoder for this batch
                writer.add_scalar('train/avg_norm/decoder', avg_norm_decoder, batch_count)
                # Determine the maximum gradient norm across decoder
                max_norm_decoder = get_max_gradient_norm(model.decoder)
                # Log the maximum gradient norm of the decoder for this batch
                writer.add_scalar('train/max_norm/decoder', max_norm_decoder, batch_count)

                # Apply gradient clipping for training stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

                # Perform an optimization step
                optimizer.step()

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
                    # Validate the model checkpoint on each validation dataset
                    validation_results[val_set.name()] = evaluate(model=model,
                                                                  eval_set=val_set,
                                                                  multipliers=multipliers,
                                                                  writer=writer,
                                                                  i=batch_count,
                                                                  device=device,
                                                                  eq_fn=eq_fn,
                                                                  eq_kwargs=eq_kwargs,
                                                                  gm_kwargs=gm_kwargs)

                # Make sure model is on correct device and switch to training mode
                model = model.to(device)
                model.train()

                if cudnn_benchmarking:
                    # Re-enable benchmarking after validation
                    torch.backends.cudnn.benchmark = True

                if decay_scheduler.patience and not warmup_scheduler.is_active() and i >= n_epochs_late_start:
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
        # Evaluate the model using testing split
        final_results = evaluate(model=best_model,
                                 eval_set=eval_set,
                                 multipliers=multipliers,
                                 device=device,
                                 eq_fn=eq_fn,
                                 eq_kwargs=eq_kwargs,
                                 gm_kwargs=gm_kwargs)

        # Log the evaluation results for this dataset in metrics.json
        ex.log_scalar(f'Evaluation Results ({eval_set.name()})', final_results, best_model_checkpoint)
