# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from timbre_trap.datasets.MixedMultiPitch import URMP as URMP_Mixtures, Bach10 as Bach10_Mixtures, Su, MusicNet, TRIOS
from timbre_trap.datasets.SoloMultiPitch import URMP as URMP_Stems, MedleyDB_Pitch, MAESTRO, GuitarSet
from timbre_trap.datasets.AudioMixtures import MedleyDB as MedleyDB_Mixtures, FMA
from timbre_trap.datasets.AudioStems import MedleyDB as MedleyDB_Stems
from timbre_trap.datasets import ComboDataset, constants

from ss_mpe.datasets.SoloMultiPitch import NSynth, SWD
from ss_mpe.datasets.AudioMixtures import MagnaTagATune
from ss_mpe.datasets import collate_audio_only

from ss_mpe.models import DataParallel, SS_MPE, TT_Base
from ss_mpe.models.objectives import *
from ss_mpe.models.utils import *
from evaluate import evaluate
from utils import *

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
    max_epochs = 500

    # Number of iterations between checkpoints
    checkpoint_interval = 250

    # Number of samples to gather for a batch
    batch_size = 4 if DEBUG else 16

    # Number of seconds of audio per sample
    n_secs = 10

    # Initial learning rate
    learning_rate = 1e-3

    # Scaling factors for each loss term
    multipliers = {
        'support' : 1,
        'harmonic' : 1,
        'sparsity' : 1,
        'timbre' : 0,
        'geometric' : 0,
        #'superposition' : 0,
        #'scaling' : 0,
        #'power' : 0
    }

    # Number of epochs spanning warmup phase (0 to disable)
    n_epochs_warmup = 1

    # Set validation dataset to compare for learning rate decay and early stopping
    validation_criteria_set = URMP_Mixtures.name()

    # Set validation metric to compare for learning rate decay and early stopping
    validation_criteria_metric = 'loss/total'

    # Select whether the validation criteria should be maximized or minimized
    validation_criteria_maximize = False # (False - minimize | True - maximize)

    # Late starting point (0 to disable)
    n_epochs_late_start = 0

    # Number of epochs without improvement before reducing learning rate (0 to disable)
    n_epochs_decay = 1

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
        root_dir = os.path.join('/', 'storage', 'frank', 'self-supervised-pitch', EX_NAME)
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
def train_model(checkpoint_path, max_epochs, checkpoint_interval, batch_size, n_secs, learning_rate, multipliers,
                n_epochs_warmup, validation_criteria_set, validation_criteria_metric, validation_criteria_maximize,
                n_epochs_late_start, n_epochs_decay, n_epochs_cooldown, n_epochs_early_stop, gpu_ids, seed,
                sample_rate, hop_length, fmin, bins_per_octave, n_bins, harmonics, n_workers, root_dir):
    # Discard read-only types
    multipliers = dict(multipliers)
    harmonics = list(harmonics)
    gpu_ids = list(gpu_ids)

    # Seed everything with the same seed
    seed_everything(seed)

    # Point to the datasets within the storage drive containing them or use the default location
    nsynth_base_dir    = os.path.join('/', 'storageNVME', 'frank', 'NSynth') if CONFIG else None
    mnet_base_dir      = os.path.join('/', 'storageNVME', 'frank', 'MusicNet') if CONFIG else None
    mydb_base_dir      = os.path.join('/', 'storage', 'frank', 'MedleyDB') if CONFIG else None
    magna_base_dir     = os.path.join('/', 'storageNVME', 'frank', 'MagnaTagATune') if CONFIG else None
    fma_base_dir       = os.path.join('/', 'storageNVME', 'frank', 'FMA') if CONFIG else None
    mydb_ptch_base_dir = os.path.join('/', 'storage', 'frank', 'MedleyDB-Pitch') if CONFIG else None
    urmp_base_dir      = os.path.join('/', 'storage', 'frank', 'URMP') if CONFIG else None
    bch10_base_dir     = os.path.join('/', 'storage', 'frank', 'Bach10') if CONFIG else None
    gset_base_dir      = os.path.join('/', 'storage', 'frank', 'GuitarSet') if CONFIG else None
    mstro_base_dir     = os.path.join('/', 'storage', 'frank', 'MAESTRO') if CONFIG else None
    swd_base_dir       = os.path.join('/', 'storage', 'frank', 'SWD') if CONFIG else None
    su_base_dir        = os.path.join('/', 'storage', 'frank', 'Su') if CONFIG else None
    trios_base_dir     = os.path.join('/', 'storage', 'frank', 'TRIOS') if CONFIG else None

    # Initialize the primary PyTorch device
    device = torch.device(f'cuda:{gpu_ids[0]}'
                          if torch.cuda.is_available() else 'cpu')

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

    if checkpoint_path is None:
        # Initialize autoencoder model and train from scratch
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

    # Initialize list to hold all training datasets
    all_train = list()

    # Set the URMP validation set as was defined in the MT3 paper
    urmp_val_splits = ['01', '02', '12', '13', '24', '25', '31', '38', '39']

    # Allocate remaining tracks to URMP training set
    urmp_train_splits = URMP_Mixtures.available_splits()

    for t in urmp_val_splits:
        # Remove validation track
        urmp_train_splits.remove(t)

    if DEBUG:
        # Instantiate NSynth validation split for training
        nsynth_train = NSynth(base_dir=nsynth_base_dir,
                              splits=['valid'],
                              midi_range=None,
                              sample_rate=sample_rate,
                              cqt=model.hcqt,
                              n_secs=n_secs,
                              seed=seed)
        all_train.append(nsynth_train)
    else:
        # Instantiate NSynth training split for training
        nsynth_train = NSynth(base_dir=nsynth_base_dir,
                              splits=['train'],
                              midi_range=None,
                              sample_rate=sample_rate,
                              cqt=model.hcqt,
                              n_secs=n_secs,
                              seed=seed)
        all_train.append(nsynth_train)

        # Instantiate MusicNet training split for training
        mnet_mixes_train = MusicNet(base_dir=mnet_base_dir,
                                    splits=['train'],
                                    sample_rate=sample_rate,
                                    cqt=model.hcqt,
                                    n_secs=n_secs,
                                    seed=seed)
        #all_train.append(mnet_mixes_train)

        # Instantiate FMA (large) dataset for training
        fma_train = FMA(base_dir=fma_base_dir,
                        splits=None,
                        sample_rate=sample_rate,
                        n_secs=n_secs,
                        seed=seed)
        #all_train.append(fma_train)

    # Combine all training datasets
    all_train = ComboDataset(all_train)

    # Initialize a PyTorch dataloader
    loader = DataLoader(dataset=all_train,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=n_workers,
                        collate_fn=collate_audio_only,
                        pin_memory=True,
                        drop_last=True)

    # Instantiate NSynth validation split for validation
    nsynth_val = NSynth(base_dir=nsynth_base_dir,
                        splits=['valid'],
                        n_tracks=200,
                        sample_rate=sample_rate,
                        cqt=model.hcqt,
                        seed=seed)

    # Instantiate URMP dataset mixtures for validation
    urmp_mixes_val = URMP_Mixtures(base_dir=urmp_base_dir,
                                   splits=urmp_val_splits,
                                   sample_rate=sample_rate,
                                   cqt=model.hcqt,
                                   seed=seed)

    # Instantiate TRIOS dataset for validation
    trios_val = TRIOS(base_dir=trios_base_dir,
                      splits=None,
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

    # Instantiate GuitarSet dataset for evaluation
    gset_test = GuitarSet(base_dir=gset_base_dir,
                          splits=['05'],
                          sample_rate=sample_rate,
                          cqt=model.hcqt,
                          seed=seed)

    # Add all validation datasets to a list
    validation_sets = [nsynth_val, urmp_mixes_val, trios_val, bch10_test, su_test, gset_test]

    # Add all evaluation datasets to a list
    evaluation_sets = [nsynth_test, urmp_mixes_val, trios_val, bch10_test, su_test, gset_test]

    # Initialize an optimizer for the model parameters
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Determine the amount of batches in one epoch
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

    # Determine the sequence length of training samples
    n_frames = int(n_secs * sample_rate / hop_length)

    # Define the maximum position index (geometric-invariance loss)
    max_position = 4 * n_frames

    # Define maximum time and frequency shift (geometric-invariance loss)
    max_shift_time = n_frames // 4
    max_shift_freq = 2 * bins_per_octave

    # Define time stretch boundaries (geometric-invariance loss)
    min_stretch_time = 0.5
    max_stretch_time = 2

    # Compute minimum MIDI frequency of each harmonic
    fmins_midi = (fmin + 12 * torch.log2(torch.Tensor(harmonics))).unsqueeze(-1)
    # Determine center MIDI frequency of each bin in the HCQT (timbre-invariance loss)
    fbins_midi = fmins_midi + torch.arange(n_bins) / (bins_per_octave / 12)

    # Loop through epochs
    for i in range(max_epochs):
        #t = get_current_time()
        # Loop through batches of audio
        for data in tqdm(loader, desc=f'Epoch {i + 1}'):
            #print_time_difference(t, 'Load Batch', device=device)
            #t = get_current_time()
            # Increment the batch counter
            batch_count += 1

            if warmup_scheduler.is_active():
                # Step the learning rate warmup scheduler
                warmup_scheduler.step()

            # Extract audio and add to appropriate device
            audio = data[constants.KEY_AUDIO].to(device)

            # Log the current learning rate for this batch
            writer.add_scalar('train/loss/learning_rate', optimizer.param_groups[0]['lr'], batch_count)
            #print_time_difference(t, 'Step/Audio', device=device)
            #t = get_current_time()

            # Compute full set of spectral features
            features = model.get_all_features(audio)

            # Extract relevant feature sets
            features_log   = features['dec']
            features_log_1 = features['dec_1']
            features_log_h = features['dec_h']
            #print_time_difference(t, 'Features', device=device)
            #t = get_current_time()

            with torch.autocast(device_type=f'cuda'):
                # Process features to obtain logits
                logits, _, losses = model(features_log)
                # Convert to (implicit) pitch salience activations
                estimate = torch.sigmoid(logits)
                #print_time_difference(t, 'Model', device=device)
                #t = get_current_time()

                # Compute support loss w.r.t. first harmonic for the batch
                support_loss = compute_support_loss(logits, features_log_1)
                # Log the support loss for this batch
                writer.add_scalar('train/loss/support', support_loss.item(), batch_count)

                # Compute harmonic loss w.r.t. weighted harmonic sum for the batch
                harmonic_loss = compute_harmonic_loss(logits, features_log_h)
                # Log the harmonic loss for this batch
                writer.add_scalar('train/loss/harmonic', harmonic_loss.item(), batch_count)

                # Compute sparsity loss for the batch
                sparsity_loss = compute_sparsity_loss(estimate)
                # Log the sparsity loss for this batch
                writer.add_scalar('train/loss/sparsity', sparsity_loss.item(), batch_count)

                # Compute timbre-invariance loss for the batch
                timbre_loss = compute_timbre_loss(model, features_log, logits, fbins_midi, bins_per_octave)
                # Log the timbre-invariance loss for this batch
                writer.add_scalar('train/loss/timbre', timbre_loss.item(), batch_count)

                """
                # Compute geometric-invariance loss for the batch
                geometric_loss = compute_geometric_loss(model, features_log, logits, max_seq_idx=max_position,
                                                        max_shift_f=max_shift_freq, max_shift_t=max_shift_time,
                                                        min_stretch=min_stretch_time, max_stretch=max_stretch_time)
                # Log the geometric-invariance loss for this batch
                writer.add_scalar('train/loss/geometric', geometric_loss.item(), batch_count)
                """

                # Compute the total loss for this batch
                total_loss = multipliers['support'] * support_loss + \
                             multipliers['harmonic'] * harmonic_loss + \
                             multipliers['sparsity'] * sparsity_loss + \
                             multipliers['timbre'] * timbre_loss# + \
                             #multipliers['geometric'] * geometric_loss

                if i >= n_epochs_late_start:
                    # Currently no late-state losses
                    pass

                for key_loss, val_loss in losses.items():
                    # Log the model loss for this batch
                    writer.add_scalar(f'train/loss/{key_loss}', val_loss.item(), batch_count)
                    # Add the model loss to the total loss
                    total_loss += multipliers.get(key_loss, 1) * val_loss

                # Log the total loss for this batch
                writer.add_scalar('train/loss/total', total_loss.item(), batch_count)
                #print_time_difference(t, 'Losses', device=device)
                #t = get_current_time()

                # Zero the accumulated gradients
                optimizer.zero_grad()
                #print_time_difference(t, 'Zero Grad', device=device)
                #t = get_current_time()
                # Compute gradients using total loss
                total_loss.backward()
                #print_time_difference(t, 'Backward', device=device)
                #t = get_current_time()

                # Track the gradient norm of each layer in the encoder
                cum_norm_encoder = track_gradient_norms(model.encoder)
                # Log the cumulative gradient norm of the encoder for this batch
                writer.add_scalar('train/cum_norm/encoder', cum_norm_encoder, batch_count)

                # Track the gradient norm of each layer in the decoder
                cum_norm_decoder = track_gradient_norms(model.decoder)
                # Log the cumulative gradient norm of the decoder for this batch
                writer.add_scalar('train/cum_norm/decoder', cum_norm_decoder, batch_count)

                # Apply gradient clipping for training stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                #print_time_difference(t, 'Track Grad', device=device)
                #t = get_current_time()

                # Perform an optimization step
                optimizer.step()
                #print_time_difference(t, 'Step', device=device)

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
                                                                  device=device)

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
            #t = get_current_time()

        if early_stop_criteria:
            # Stop training
            break

    print(f'Achieved best results at {best_model_checkpoint} iterations...')

    for val_set in validation_sets:
        # Log the results at the best checkpoint for each validation dataset in metrics.json
        ex.log_scalar(f'Validation Results ({val_set.name()})', best_results[val_set.name()], best_model_checkpoint)

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
                                 device=device)

        # Log the evaluation results for this dataset in metrics.json
        ex.log_scalar(f'Evaluation Results ({eval_set.name()})', final_results, best_model_checkpoint)
