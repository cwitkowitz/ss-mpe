# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from timbre_trap.datasets.MixedMultiPitch import (URMP as URMP_Mixtures,
                                                  Bach10 as Bach10_Mixtures,
                                                  MusicNet,
                                                  Su,
                                                  TRIOS)
from timbre_trap.datasets.SoloMultiPitch import (MAESTRO,
                                                 GuitarSet,
                                                 MedleyDB_Pitch)
from timbre_trap.datasets.AudioMixtures import (FMA,
                                                MedleyDB)
from timbre_trap.datasets import ComboDataset

from ss_mpe.datasets.SoloMultiPitch import (NSynth,
                                            SWD,
                                            MIR_1K)
from ss_mpe.datasets.AudioMixtures import (E_GMD,
                                           MagnaTagATune)

from ss_mpe.framework import SS_MPE, TT_Base, TT_Enc
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


CONFIG = 0 # (0 - desktop | 1 - lab)
EX_NAME = '_'.join(['ScaleUp_EG'])

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

    # Initial learning rate for encoder
    learning_rate_encoder = 1e-3

    # Initial learning rate for decoder
    learning_rate_decoder = learning_rate_encoder

    # Group together both learning rates
    learning_rates = [learning_rate_encoder, learning_rate_decoder]

    # Scaling factors for each loss term
    multipliers = {
        'support' : 1,
        'harmonic' : 1,
        'sparsity' : 1,
        'timbre' : 0,
        'geometric' : 0,
        'percussion' : 0,
        'supervised' : 0
    }

    # Whether to formulate objectives as self-supervision or augmentation
    self_supervised_targets = True

    # Number of epochs spanning warmup phase (0 to disable)
    n_epochs_warmup = 5

    # Set validation dataset to compare for learning rate decay and early stopping
    validation_criteria_set = URMP_Mixtures.name()

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
    gpu_ids = [1, 0]

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
    n_workers = 0 #8 * len(gpu_ids)

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
def train_model(checkpoint_path, max_epochs, checkpoint_interval, batch_size, n_secs, learning_rates, multipliers,
                self_supervised_targets, n_epochs_warmup, validation_criteria_set, validation_criteria_metric,
                validation_criteria_maximize, n_epochs_decay, n_epochs_cooldown, n_epochs_early_stop, gpu_ids,
                seed, sample_rate, hop_length, fmin, bins_per_octave, n_bins, harmonics, n_workers, root_dir):
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

    # TODO - consider alternate harmonics
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
                        model_complexity=2,
                        skip_connections=False)
        # Initialize Timbre-Trap encoder
        #model = TT_Enc(hcqt_params,
        #               model_complexity=2)
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
    fma_base_dir    = os.path.join('/', 'storageNVME', 'frank', 'FMA') if CONFIG else None
    mdb_base_dir    = os.path.join('/', 'storage', 'frank', 'MedleyDB') if CONFIG else None
    magna_base_dir  = os.path.join('/', 'storageNVME', 'frank', 'MagnaTagATune') if CONFIG else None
    egmd_base_dir   = os.path.join('/', 'storageNVME', 'frank', 'E-GMD') if CONFIG else None
    nsynth_base_dir = os.path.join('/', 'storageNVME', 'frank', 'NSynth') if CONFIG else None

    # MPE dataset paths
    urmp_base_dir     = os.path.join('/', 'storage', 'frank', 'URMP') if CONFIG else None
    bch10_base_dir    = os.path.join('/', 'storageNVME', 'frank', 'Bach10') if CONFIG else None
    gset_base_dir     = os.path.join('/', 'storageNVME', 'frank', 'GuitarSet') if CONFIG else None
    mdb_ptch_base_dir = os.path.join('/', 'storage', 'frank', 'MedleyDB-Pitch') if CONFIG else None

    # AMT dataset paths
    mstro_base_dir = os.path.join('/', 'storageNVME', 'frank', 'MAESTRO') if CONFIG else None
    mnet_base_dir  = os.path.join('/', 'storageNVME', 'frank', 'MusicNet') if CONFIG else None
    swd_base_dir   = os.path.join('/', 'storage', 'frank', 'SWD') if CONFIG else None
    su_base_dir    = os.path.join('/', 'storageNVME', 'frank', 'Su') if CONFIG else None
    trios_base_dir = os.path.join('/', 'storageNVME', 'frank', 'TRIOS') if CONFIG else None

    # Initialize list to hold training datasets
    all_train = list()

    # Instantiate NSynth training split for training
    nsynth_train = NSynth(base_dir=nsynth_base_dir,
                          splits=['train'],
                          midi_range=np.array([fmin, fmax]),
                          sample_rate=sample_rate,
                          n_secs=n_secs,
                          seed=seed)
    all_train.append(nsynth_train)

    # Instantiate MusicNet audio (training) mixtures for training
    mnet_audio = MusicNet(base_dir=mnet_base_dir,
                          splits=['train'],
                          sample_rate=sample_rate,
                          n_secs=n_secs,
                          seed=seed)
    #all_train.append(mnet_audio)

    # Define mostly-harmonic splits for FMA
    fma_genres_harmonic = ['Rock', 'Folk', 'Instrumental', 'Pop', 'Classical','Jazz', 'Country', 'Soul-RnB', 'Blues']

    # Instantiate FMA audio mixtures for training
    fma_audio = FMA(base_dir=fma_base_dir,
                    splits=fma_genres_harmonic,
                    sample_rate=sample_rate,
                    n_secs=n_secs,
                    seed=seed)
    #all_train.append(fma_audio)

    # Instantiate MedleyDB audio mixtures for training
    mdb_audio = MedleyDB(base_dir=mdb_base_dir,
                         sample_rate=sample_rate,
                         n_secs=n_secs,
                         seed=seed)
    #all_train.append(mdb_audio)

    # Combine training datasets
    all_train = ComboDataset(all_train)

    # Initialize a PyTorch dataloader for data
    loader = DataLoader(dataset=all_train,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=n_workers,
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

    # Set the URMP validation set in accordance with the MT3 paper
    urmp_val_splits = ['01', '02', '12', '13', '24', '25', '31', '38', '39']

    # Instantiate URMP dataset mixtures for validation
    urmp_val = URMP_Mixtures(base_dir=urmp_base_dir,
                             splits=urmp_val_splits,
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

    # Instantiate TRIOS dataset for evaluation
    trios_test = TRIOS(base_dir=trios_base_dir,
                       splits=None,
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

    # Add all validation datasets to a list
    validation_sets = [nsynth_val, urmp_val, bch10_test, su_test, trios_test, gset_val]

    # Add all evaluation datasets to a list
    evaluation_sets = [bch10_test, su_test, trios_test, gset_test]

    #################
    ## PREPARATION ##
    #################

    # Initialize an optimizer for the model parameters with differential learning rates
    optimizer = torch.optim.AdamW([{'params' : model.encoder_parameters(), 'lr' : learning_rates[0]},
                                   {'params' : model.decoder_parameters(), 'lr' : learning_rates[1]}])

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

    # Number of random points to sample per octave
    points_per_octave = 3

    # Infer the number of bins per semitone
    bins_per_semitone = bins_per_octave / 12

    # Determine semitone span of frequency support
    semitone_span = model.hcqt.midi_freqs.max() - model.hcqt.midi_freqs.min()

    # Determine how many bins are represented across all harmonics
    n_psuedo_bins = (bins_per_semitone * semitone_span).round()

    # Determine how many octaves have been covered
    n_octaves = int(math.ceil(n_psuedo_bins / bins_per_octave))

    # Calculate number of cut/boost points to sample
    n_points = 1 + points_per_octave * n_octaves

    # Standard deviation of boost/cut
    std_dev = 0.25

    # Set keyword arguments for random equalization
    random_kwargs = {
        'n_points' : n_points,
        'std_dev' : std_dev
    }

    # Set equalization type and corresponding parameter values
    eq_fn, eq_kwargs = sample_random_equalization, random_kwargs

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

    #####################
    ## PERCUSSION LOSS ##
    #####################

    # Initialize list to hold unpitched datasets
    perc_train = list()

    # Instantiate E-GMD audio for percussion-invariance
    egmd = E_GMD(base_dir=egmd_base_dir,
                 splits=['train'],
                 sample_rate=sample_rate,
                 n_secs=n_secs,
                 seed=seed)
    perc_train.append(egmd)

    # TODO - include other noise / percussion datasets

    # Combine unpitched datasets
    perc_train = ComboDataset(perc_train)

    # Initialize a PyTorch dataloader for percussion data
    loader_pc = DataLoader(dataset=perc_train,
                           batch_size=batch_size,
                           shuffle=True,
                           num_workers=n_workers,
                           pin_memory=True,
                           drop_last=True)

    ##############################
    ## TRAINING/VALIDATION LOOP ##
    ##############################

    # Loop through epochs
    for i in range(max_epochs):
        # Loop through batches of both types of data
        for data in tqdm(loader, desc=f'Epoch {i + 1}'):
            # Increment the batch counter
            batch_count += 1

            # Extract audio and add audio to appropriate device
            audio = data[constants.KEY_AUDIO].to(device)
            # Extract ground-truth pitch salience activations
            #ground_truth = data[constants.KEY_GROUND_TRUTH].to(device)

            # Log the current learning rates for this batch
            writer.add_scalar('train/loss/learning_rate/encoder', optimizer.param_groups[0]['lr'], batch_count)
            writer.add_scalar('train/loss/learning_rate/decoder', optimizer.param_groups[1]['lr'], batch_count)

            # Compute full set of spectral features
            features = model.get_all_features(audio)

            # Extract relevant feature sets
            features_db   = features['db']
            features_db_1 = features['db_1']
            features_db_h = features['db_h']

            # Sample a batch of percussion audio
            data_pc = next(iter(loader_pc))
            # Sample random volumes for percussion audio
            volumes = 2.0 * torch.rand((batch_size, 1, 1), device=device)
            # Superimpose percussion audio onto original audio
            audio_pc = audio + volumes * data_pc[constants.KEY_AUDIO].to(device)
            # Compute spectral features for percussion audio mixture
            features_pc = model.hcqt.to_decibels(model.hcqt(audio_pc))

            with torch.autocast(device_type=f'cuda'):
                # Process features to obtain logits
                logits, _, _ = model(features_db)
                # Convert to (implicit) pitch salience activations
                activations = torch.sigmoid(logits)

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
                sparsity_loss = compute_sparsity_loss(activations)
                # Log the sparsity loss for this batch
                writer.add_scalar('train/loss/sparsity', sparsity_loss.item(), batch_count)

                debug_nans(sparsity_loss, 'sparsity')

                # Determine whether targets should be logits or ground-truth
                targets = activations #if self_supervised_targets else ground_truth

                # Compute timbre-invariance loss for the batch
                timbre_loss = compute_timbre_loss(model, features_db, targets, eq_fn=eq_fn, **eq_kwargs)
                # Log the timbre-invariance loss for this batch
                writer.add_scalar('train/loss/timbre', timbre_loss.item(), batch_count)

                debug_nans(timbre_loss, 'timbre')

                # Compute geometric-equivariance loss for the batch
                geometric_loss = compute_geometric_loss(model, features_db, targets, **gm_kwargs)
                # Log the geometric-equivariance loss for this batch
                writer.add_scalar('train/loss/geometric', geometric_loss.item(), batch_count)

                debug_nans(geometric_loss, 'geometric')

                # Compute percussion-invariance loss for the batch
                percussion_loss = compute_percussion_loss(model, features_pc, targets)
                # Log the percussion-invariance loss for this batch
                writer.add_scalar('train/loss/percussion', percussion_loss.item(), batch_count)

                debug_nans(percussion_loss, 'percussion')

                # Compute supervised BCE loss for the batch
                #supervised_loss = compute_supervised_loss(logits, ground_truth, True)
                # Log the supervised BCE loss for this batch
                #writer.add_scalar('train/loss/supervised', supervised_loss.item(), batch_count)

                #debug_nans(supervised_loss, 'supervised')

                # Compute the total loss for this batch
                total_loss = multipliers['support'] * support_loss + \
                             multipliers['harmonic'] * harmonic_loss + \
                             multipliers['sparsity'] * sparsity_loss + \
                             multipliers['timbre'] * timbre_loss + \
                             multipliers['geometric'] * geometric_loss + \
                             multipliers['percussion'] * percussion_loss# + \
                             #multipliers['supervised'] * supervised_loss

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
                                                                      self_supervised_targets=self_supervised_targets,
                                                                      eq_fn=eq_fn,
                                                                      eq_kwargs=eq_kwargs,
                                                                      gm_kwargs=gm_kwargs,
                                                                      pc_set=egmd)
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
                                     device=device,
                                     self_supervised_targets=self_supervised_targets,
                                     eq_fn=eq_fn,
                                     eq_kwargs=eq_kwargs,
                                     gm_kwargs=gm_kwargs,
                                     pc_set=egmd)
        except Exception as e:
            print(f'Error evaluating \'{eval_set.name()}\': {repr(e)}')

        # Log the evaluation results for this dataset in metrics.json
        ex.log_scalar(f'Evaluation Results ({eval_set.name()})', final_results, best_model_checkpoint)
