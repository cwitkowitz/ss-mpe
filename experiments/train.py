# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from timbre_trap.datasets.MixedMultiPitch import URMP as URMP_Mixtures, Bach10 as Bach10_Mixtures, Su, MusicNet, TRIOS
from timbre_trap.datasets.SoloMultiPitch import URMP as URMP_Stems, MedleyDB_Pitch, MAESTRO, GuitarSet
from timbre_trap.datasets.AudioStems import MedleyDB as MedleyDB_Stems
from timbre_trap.datasets.AudioMixtures import MedleyDB as MedleyDB_Mixtures, FMA
from timbre_trap.datasets import ComboDataset, constants
from timbre_trap.models import DataParallel

from ss_mpe.datasets.SoloMultiPitch import NSynth, SWD
from ss_mpe.datasets.AudioMixtures import MagnaTagATune

from ss_mpe.models.hcqt import LHVQT
from ss_mpe.models.model import SA_UNet
from ss_mpe.models.objectives import *
from lhvqt import torch_amplitude_to_db

from evaluate import evaluate
from utils import *

# Regular imports
from torch.utils.tensorboard import SummaryWriter
from sacred.observers import FileStorageObserver
from torch.utils.data import DataLoader
from sacred import Experiment
from tqdm import tqdm

import librosa
import torch
import os


SYNTH = 1 # (0 FMA | 1 - NSynth)
CONFIG = 0 # (0 - desktop | 1 - lab)
EX_NAME = '_'.join(['<EXPERIMENT_NAME>'])

ex = Experiment('Train a model to learn representations for MPE')


@ex.config
def config():
    ##############################
    ## TRAINING HYPERPARAMETERS ##
    ##############################

    # Maximum number of training iterations to conduct
    max_epochs = 50 if SYNTH else 10

    # Number of iterations between checkpoints
    checkpoint_interval = 50

    # Number of samples to gather for a batch
    #batch_size = (150 if CONFIG else 50) if SYNTH else (24 if CONFIG else 4)
    batch_size = (150 if CONFIG else 50) if SYNTH else (16 if CONFIG else 4)

    # Number of seconds of audio per sample
    n_secs = 4 if SYNTH else 30

    # Fixed learning rate
    learning_rate = 1e-3

    # Scaling factors for each loss term
    multipliers = {
        'power' : 10,
        'support' : 1,
        'harmonic' : 1,
        'sparsity' : 1,
        'timbre' : 1,
        'geometric' : 0,
        'superposition' : 0,
        'scaling' : 0
    }

    # IDs of the GPUs to use, if available
    #gpu_ids = [0, 1, 2] if CONFIG else [0]
    gpu_ids = [1, 2] if CONFIG else [0]

    # Random seed for this experiment
    seed = 0

    ########################
    ## FEATURE EXTRACTION ##
    ########################

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

    ############
    ## OTHERS ##
    ############

    # Switch for managing multiple path layouts (0 - local | 1 - lab)
    path_layout = 1 if CONFIG else 0

    # Number of threads to use for data loading
    n_workers = (16 if CONFIG else 0) if SYNTH else (8 if CONFIG else 0)

    if path_layout:
        root_dir = os.path.join('/', 'storage', 'frank', 'self-supervised-pitch', EX_NAME)
    else:
        root_dir = os.path.join('.', 'generated', 'experiments', EX_NAME)

    # Create the root directory for the experiment files
    os.makedirs(root_dir, exist_ok=True)

    # Add a file storage observer for the log directory
    ex.observers.append(FileStorageObserver(root_dir))


@ex.automain
def train_model(max_epochs, checkpoint_interval, batch_size, n_secs,
                learning_rate, multipliers, gpu_ids, seed, sample_rate,
                hop_length, n_bins, bins_per_octave, harmonics, fmin,
                path_layout, n_workers, root_dir):
    # Discard read-only types
    multipliers = dict(multipliers)
    harmonics = list(harmonics)

    # Seed everything with the same seed
    seed_everything(seed)

    # Determine the sequence length of training samples
    n_frames = int(n_secs * sample_rate / hop_length)

    # Determine index of first harmonic (support/content loss)
    h_idx = harmonics.index(1)

    # Create weighting for harmonics (harmonic loss)
    harmonic_weights = 1 / torch.Tensor(harmonics) ** 2
    # Apply zero weight to sub-harmonics (harmonic loss)
    harmonic_weights[harmonic_weights > 1] = 0
    # Normalize the harmonic weights
    harmonic_weights /= torch.sum(harmonic_weights)
    # Add frequency and time dimensions for broadcasting
    harmonic_weights = harmonic_weights.unsqueeze(-1).unsqueeze(-1)

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

    # Define probability of mixing two tracks in a batch (superposition loss)
    mix_probability = np.log2(batch_size) / batch_size

    if path_layout:
        # Point to the storage drives containing each dataset
        fma_base_dir = os.path.join('/', 'storageNVME', 'frank', 'FreeMusicArchive')
        nsynth_base_dir = os.path.join('/', 'storageNVME', 'frank', 'NSynth')
        bach10_base_dir = os.path.join('/', 'storage', 'frank', 'Bach10')
        su_base_dir = os.path.join('/', 'storage', 'frank', 'Su')
        trios_base_dir = os.path.join('/', 'storage', 'frank', 'TRIOS')
    else:
        # Use the default base directory paths
        fma_base_dir = None
        nsynth_base_dir = None
        bach10_base_dir = None
        su_base_dir = None
        trios_base_dir = None

    # Instantiate FreeMusicArchive dataset for training
    freemusicarchive = FreeMusicArchive(base_dir=fma_base_dir,
                                        sample_rate=sample_rate,
                                        n_secs=n_secs,
                                        seed=seed)

    # Instantiate NSynth dataset for training
    nsynthtrain = NSynth(base_dir=nsynth_base_dir,
                         splits=['train'],
                         sample_rate=sample_rate,
                         n_secs=n_secs,
                         seed=seed)

    # Combine all training datasets into one
    training_data = ComboSet([nsynthtrain]) if SYNTH else ComboSet([freemusicarchive])

    # Determine maximum supported MIDI frequency
    fmax = max(fbins_midi[h_idx]).item()

    # Instantiate NSynth dataset for validation
    nsynthvalid = NSynthValidation(base_dir=nsynth_base_dir,
                                   splits=['valid'],
                                   n_tracks=150 if CONFIG else 50,
                                   midi_range=[fmin, fmax],
                                   sample_rate=sample_rate)
    #training_data.datasets[0].tracks = nsynthvalid.tracks

    # Instantiate Bach10 dataset for validation
    bach10 = Bach10(base_dir=bach10_base_dir,
                    sample_rate=sample_rate)

    # Instantiate Su dataset for validation
    su = Su(base_dir=su_base_dir,
            sample_rate=sample_rate)

    # Instantiate TRIOS dataset for validation
    trios = TRIOS(base_dir=trios_base_dir,
                  sample_rate=sample_rate)

    # Initialize a list to hold all validation datasets
    validation_sets = [nsynthvalid, bach10, su, trios]

    # Initialize a PyTorch dataloader for the data
    loader = DataLoader(dataset=training_data,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=n_workers,
                        drop_last=True)

    # Initialize the HCQT feature extraction module
    hcqt = LHVQT(fs=sample_rate,
                 hop_length=hop_length,
                 fmin=librosa.midi_to_hz(fmin),
                 n_bins=n_bins,
                 bins_per_octave=bins_per_octave,
                 harmonics=harmonics,
                 update=False,
                 to_db=False,
                 db_to_prob=False,
                 batch_norm=False)

    # Initialize MPE representation learning model
    model = SA_UNet(n_ch_in=len(harmonics),
                   n_bins_in=n_bins,
                   model_complexity=2)

    # Initialize the primary PyTorch device
    device = torch.device(f'cuda:{gpu_ids[0]}'
                          if torch.cuda.is_available() else 'cpu')

    if len(gpu_ids) > 1:
        # Wrap feature extraction and model for multi-GPU usage
        hcqt = torch.nn.DataParallel(hcqt, device_ids=gpu_ids)
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)

    # Add model and feature extraction to primary device
    hcqt, model = hcqt.to(device), model.to(device)

    # Make sure weights are on appropriate device
    harmonic_weights = harmonic_weights.to(device)

    # Initialize an optimizer for the model parameters
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Construct the path to the directory for saving models
    log_dir = os.path.join(root_dir, 'models')

    # Initialize a writer to log results
    writer = SummaryWriter(log_dir)

    # Number of batches that have been processed
    batch_count = 0

    # Determine the amount of batches in one epoch
    epoch_steps = len(loader)

    # Loop through epochs
    for i in range(max_epochs):
        # Loop through batches
        for audio in tqdm(loader, desc=f'Epoch {i + 1}'):
            # Add audio to the appropriate device
            audio = audio.to(device)

            with torch.autocast(device_type=f'cuda'):
                # Obtain spectral features in decibels
                features_dec = torch_amplitude_to_db(hcqt(audio))
                # Convert decibels to linear gain between 0 and 1
                features_lin = decibels_to_amplitude(features_dec)
                # Scale decibels to be between 0 and 1
                features_log = rescale_decibels(features_dec)

                # Compute pitch salience embeddings
                embeddings = model(features_log).squeeze()

                # Convert logits to activations (implicit pitch salience)
                salience = torch.sigmoid(embeddings)

                # Obtain pseudo-ground-truth as features at first harmonic
                pseudo_ground_truth_lin = features_lin[:, h_idx]
                pseudo_ground_truth_log = features_log[:, h_idx]

                # Compute a weighted sum of the features to obtain a rough salience estimate
                pseudo_salience_lin = torch.sum(features_lin * harmonic_weights, dim=-3)
                pseudo_salience_log = torch.sum(features_log * harmonic_weights, dim=-3)
                #pseudo_salience = torch.Tensor(filter_non_peaks(pseudo_salience.cpu().numpy())).to(device)

                # Compute the power loss for this batch
                power_loss = compute_power_loss(salience, pseudo_ground_truth_log)

                # Log the power loss for this batch
                writer.add_scalar('train/loss/power', power_loss, batch_count)

                # Compute the support loss with respect to the first harmonic for this batch
                support_loss = compute_support_loss(embeddings, pseudo_ground_truth_log)

                # Log the support loss for this batch
                writer.add_scalar('train/loss/support', support_loss, batch_count)

                # Compute the harmonic loss with respect to the weighted harmonic sum for this batch
                harmonic_loss = compute_harmonic_loss(embeddings, pseudo_salience_log)

                # Log the harmonic loss for this batch
                writer.add_scalar('train/loss/harmonic', harmonic_loss, batch_count)

                # Compute the sparsity loss for this batch
                sparsity_loss = compute_sparsity_loss(salience)

                # Log the sparsity loss for this batch
                writer.add_scalar('train/loss/sparsity', sparsity_loss, batch_count)

                # Compute the timbre-invariance loss for this batch
                timbre_loss = compute_timbre_loss(model, features_log, embeddings, fbins_midi, bins_per_octave)

                # Log the timbre-invariance loss for this batch
                writer.add_scalar('train/loss/timbre', timbre_loss, batch_count)

                # Compute the geometric-invariance loss for this batch
                #geometric_loss = compute_geometric_loss(model, features_log, embeddings, max_seq_idx=max_position,
                #                                        max_shift_f=max_shift_freq, max_shift_t=max_shift_time,
                #                                        min_stretch=min_stretch_time, max_stretch=max_stretch_time)

                # Log the geometric-invariance loss for this batch
                #writer.add_scalar('train/loss/geometric', geometric_loss, batch_count)

                # Compute the superposition loss for this batch
                #superposition_loss = compute_superposition_loss(hcqt, model, audio, salience, mix_probability)

                # Log the superposition loss for this batch
                #writer.add_scalar('train/loss/superposition', superposition_loss, batch_count)

                # Compute the scaling loss for this batch
                #scaling_loss = compute_scaling_loss(model, features_log, salience)

                # Log the scaling loss for this batch
                #writer.add_scalar('train/loss/scaling', scaling_loss, batch_count)

                # Compute the total loss for this batch
                loss = multipliers['power'] * power_loss + \
                       multipliers['support'] * support_loss * cosine_anneal(batch_count, epoch_steps, floor=0.) + \
                       multipliers['harmonic'] * harmonic_loss * cosine_anneal(batch_count, epoch_steps, floor=0.) + \
                       multipliers['sparsity'] * sparsity_loss * (1 - cosine_anneal(batch_count, epoch_steps, floor=0.)) + \
                       multipliers['timbre'] * timbre_loss
                       #multipliers['geometric'] * geometric_loss + \
                       #multipliers['superposition'] * superposition_loss + \
                       #multipliers['scaling'] * scaling_loss

                # Log the total loss for this batch
                writer.add_scalar('train/loss/total', loss, batch_count)

                # Zero the accumulated gradients
                optimizer.zero_grad()
                # Compute gradients based on total loss
                loss.backward()
                # Perform an optimization step
                optimizer.step()

            # Increment the batch counter
            batch_count += 1

            if batch_count % checkpoint_interval == 0:
                for val_set in validation_sets:
                    # Validate the model with each validation dataset
                    results = evaluate(model=model,
                                       hcqt=hcqt,
                                       eval_set=val_set,
                                       writer=writer,
                                       i=batch_count,
                                       device=device)

                # Place model back in training mode
                model.train()

                # Construct a path to save the model checkpoint
                model_path = os.path.join(log_dir, f'model-{batch_count}.pt')

                if isinstance(model, torch.nn.DataParallel):
                    # Unwrap and save the model
                    torch.save(model.module, model_path)
                else:
                    # Save the model as is
                    torch.save(model, model_path)

                # TODO - add stopping criterion here
