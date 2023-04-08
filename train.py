# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from common import ComboSet
from FreeMusicArchive import FreeMusicArchive
from NSynth import NSynth
from ToyNSynth import ToyNSynthEval
from Bach10 import Bach10
from Su import Su
from TRIOS import TRIOS
from model import SAUNet
from lhvqt import LHVQT
from objectives import *
from utils import *
from evaluate import evaluate

# Regular imports
from torch.utils.tensorboard import SummaryWriter
from sacred.observers import FileStorageObserver
from torch.utils.data import DataLoader
from sacred import Experiment
from tqdm import tqdm

import librosa
import torch
import os


CONFIG = 0 # (0 - desktop | 1 - lab)
EX_NAME = '_'.join(['Refinements'])

ex = Experiment('Train a model to learn representations for MPE')


@ex.config
def config():
    ##############################
    ## TRAINING HYPERPARAMETERS

    # Maximum number of training iterations to conduct
    max_epochs = 10

    # Number of iterations between checkpoints
    checkpoint_interval = 50

    # Number of samples to gather for a batch
    batch_size = 96 if CONFIG else 32

    # Number of seconds of audio per sample
    n_secs = 4 if CONFIG else 4

    # Fixed learning rate
    learning_rate = 1e-3

    # Scaling factors for each loss term
    multipliers = {
        'support' : 1,
        'content' : 1,
        'harmonic' : 1,
        'geometric' : 1,
        'timbre' : 1,
        'superposition' : 0
    }

    # IDs of the GPUs to use, if available
    #gpu_ids = [0, 1, 2] if CONFIG else [0]
    gpu_ids = [0, 1] if CONFIG else [0]

    # Random seed for this experiment
    seed = 0

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

    ##############################
    # OTHERS

    # Switch for managing multiple path layouts (0 - local | 1 - lab)
    path_layout = 1 if CONFIG else 0

    # Number of threads to use for data loading
    #n_workers = 8 if CONFIG else 4
    n_workers = 8 if CONFIG else 0

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

    # Define maximum time and frequency shift (geometric loss)
    max_shift_time = n_frames // 4
    max_shift_freq = 2 * bins_per_octave

    # Define time stretch boundaries (geometric loss)
    min_stretch_time = 0.5
    max_stretch_time = 2

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
        bach10_base_dir= None
        su_base_dir= None
        trios_base_dir = None

    # Instantiate FreeMusicArchive dataset for training
    freemusicarchive = FreeMusicArchive(base_dir=fma_base_dir,
                                        sample_rate=sample_rate,
                                        n_secs=n_secs,
                                        seed=seed)

    # Instantiate NSynth dataset for training
    nsynth = NSynth(base_dir=nsynth_base_dir,
                    splits=['train'],
                    sample_rate=sample_rate,
                    n_secs=n_secs,
                    seed=seed)

    # Combine all training datasets into one
    #training_data = ComboSet([freemusicarchive])
    training_data = ComboSet([nsynth])

    # Instantiate Su dataset for validation
    toynsynthtest = ToyNSynthEval(base_dir=nsynth_base_dir,
                                  sample_rate=sample_rate,
                                  hop_length=hop_length,
                                  fmin=fmin,
                                  n_bins=n_bins,
                                  bins_per_octave=bins_per_octave)

    # Instantiate Bach10 dataset for validation
    bach10 = Bach10(base_dir=bach10_base_dir,
                    sample_rate=sample_rate,
                    hop_length=hop_length,
                    fmin=fmin,
                    n_bins=n_bins,
                    bins_per_octave=bins_per_octave)

    # Instantiate Su dataset for validation
    su = Su(base_dir=su_base_dir,
            sample_rate=sample_rate,
            hop_length=hop_length,
            fmin=fmin,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave)

    # Instantiate Su dataset for validation
    trios = TRIOS(base_dir=trios_base_dir,
                  sample_rate=sample_rate,
                  hop_length=hop_length,
                  fmin=fmin,
                  n_bins=n_bins,
                  bins_per_octave=bins_per_octave)

    # Initialize a list to hold all validation datasets
    validation_sets = [toynsynthtest, bach10, su, trios]

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
                 db_to_prob=False,
                 batch_norm=False)

    # Initialize MPE representation learning model
    model = SAUNet(n_ch_in=len(harmonics),
                   n_bins_in=n_bins,
                   model_complexity=2,
                   # TODO - uncomment the following when ready to test on longer sequences
                   #max_seq=4*n_frames)
                   )

    # Initialize the primary PyTorch device
    device = torch.device(f'cuda:{gpu_ids[0]}'
                          if torch.cuda.is_available() else 'cpu')

    if len(gpu_ids) > 1:
        # Wrap feature extraction and model for multi-GPU usage
        hcqt = torch.nn.DataParallel(hcqt, device_ids=gpu_ids)
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)

    # Add model and feature extraction to primary device
    hcqt, model = hcqt.to(device), model.to(device)

    # Initialize an optimizer for the model parameters
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Construct the path to the directory for saving models
    log_dir = os.path.join(root_dir, 'models')

    # Initialize a writer to log results
    writer = SummaryWriter(log_dir)

    # Number of batches that have been processed
    batch_count = 0

    # Loop through epochs
    for i in range(max_epochs):
        # Loop through batches
        for audio in tqdm(loader, desc=f'Epoch {i}'):
            # Add audio to the appropriate device
            audio = audio.to(device)

            with torch.autocast(device_type=f'cuda'):
                # Obtain spectral features in decibels
                features_dec = hcqt(audio)
                # Convert decibels to linear gain between 0 and 1
                features_lin = decibels_to_amplitude(features_dec)
                # Scale decibels to be between 0 and 1
                features_log = rescale_decibels(features_dec)

                # Compute pitch salience embeddings
                embeddings = model(features_log).squeeze()

                # Convert logits to activations (implicit pitch salience)
                salience = torch.sigmoid(embeddings)

                # Obtain pseudo-ground-truth as features at first harmonic
                pseudo_ground_truth = features_lin[:, h_idx]

                # Compute the support loss with respect to the first harmonic for this batch
                support_loss = compute_support_loss(embeddings, pseudo_ground_truth)

                # Log the support loss for this batch
                writer.add_scalar('train/loss/support', support_loss, batch_count)

                # Compute the content loss for this batch
                content_loss = compute_content_loss(salience, pseudo_ground_truth)

                # Log the content loss for this batch
                writer.add_scalar('train/loss/content', content_loss, batch_count)

                # Compute the harmonic loss for this batch
                harmonic_loss = compute_harmonic_loss(salience, features_lin, weights=harmonic_weights)

                # Log the harmonic loss for this batch
                writer.add_scalar('train/loss/harmonic', content_loss, batch_count)

                # Compute the geometric-invariance loss for this batch
                geometric_loss = compute_geometric_loss(model, features_log, salience,
                                                        max_shift_f=max_shift_freq, max_shift_t=max_shift_time,
                                                        min_stretch=min_stretch_time, max_stretch=max_stretch_time)

                # Log the geometric-invariance loss for this batch
                writer.add_scalar('train/loss/geometric', geometric_loss, batch_count)

                # Compute the timbre-invariance loss for this batch
                #timbre_loss = compute_timbre_loss(model, embeddings, features_log, n_bins, bins_per_octave)
                timbre_loss = 0

                # Log the timbre-invariance loss for this batch
                writer.add_scalar('train/loss/timbre', timbre_loss, batch_count)

                # Compute the superposition loss for this batch
                superposition_loss = compute_superposition_loss(hcqt, model, audio, salience, mix_probability)

                # Log the superposition loss for this batch
                writer.add_scalar('train/loss/superposition', superposition_loss, batch_count)

                # Compute the total loss for this batch
                loss = multipliers['support'] * support_loss + \
                       multipliers['content'] * content_loss + \
                       multipliers['harmonic'] * harmonic_loss + \
                       multipliers['geometric'] * geometric_loss + \
                       multipliers['timbre'] * timbre_loss + \
                       multipliers['superposition'] * superposition_loss

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
