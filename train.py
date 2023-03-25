# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from common import ComboSet
from FreeMusicArchive import FreeMusicArchive
from MagnaTagATune import MagnaTagATune
from NSynth import NSynth
from Bach10 import Bach10
from model import SAUNet
from lhvqt import LHVQT
from objectives import *
from utils import *
from evaluate import evaluate

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


EX_NAME = '_'.join(['Hustlin'])

ex = Experiment('Train a model to learn representations for MPE')


@ex.config
def config():
    ##############################
    # TRAINING HYPERPARAMETERS

    # Maximum number of training iterations to conduct
    max_epochs = 1000

    # Number of iterations between checkpoints
    checkpoint_interval = 50

    # Number of samples to gather for a batch
    batch_size = 4

    # Number of seconds of audio per sample
    n_secs = 30

    # Fixed learning rate
    learning_rate = 1e-3

    # Scaling factors for each loss term
    multipliers = {
        'support' : 0,
        'content' : 2,
        'linearity' : 1,
        'invariance' : 1,
        'translation' : 1
    }

    # ID of the gpu to use, if available
    gpu_id = 0

    # Random seed for this experiment
    seed = 0

    ##############################
    # FEATURE EXTRACTION

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
    path_layout = 0  # TODO

    # Number of threads to use for data loading
    n_workers = 0

    # Create the root directory for the experiment files
    root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'generated', 'experiments', EX_NAME)
    os.makedirs(root_dir, exist_ok=True)

    # Add a file storage observer for the log directory
    ex.observers.append(FileStorageObserver(root_dir))


@ex.automain
def train_model(max_epochs, checkpoint_interval, batch_size, n_secs,
                learning_rate, multipliers, gpu_id, seed, sample_rate,
                hop_length, n_bins, bins_per_octave, harmonics, fmin,
                path_layout, n_workers, root_dir):
    # Discard read-only types
    multipliers = dict(multipliers)
    harmonics = list(harmonics)

    # Seed everything with the same seed
    seed_everything(seed)

    # Initialize a device pointer
    device = torch.device(f'cuda:{gpu_id}'
                          if torch.cuda.is_available() else 'cpu')

    # Instantiate MagnaTagATune dataset for training
    magnatagatune = MagnaTagATune(sample_rate=sample_rate,
                                  n_secs=n_secs,
                                  seed=seed)

    # Instantiate FreeMusicArchive dataset for training
    freemusicarchive = FreeMusicArchive(sample_rate=sample_rate,
                                        n_secs=n_secs,
                                        seed=seed)

    # Instantiate NSynth dataset for training
    nsynth = NSynth(sample_rate=sample_rate,
                    n_secs=n_secs,
                    seed=seed)

    # Combine all training datasets into one
    training_data = ComboSet([magnatagatune, freemusicarchive, nsynth])

    # Initialize a PyTorch dataloader for the data
    loader = DataLoader(dataset=training_data,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=n_workers,
                        drop_last=True)

    # Initialize MPE representation learning model
    model = SAUNet(n_ch_in=len(harmonics),
                   n_bins_in=n_bins,
                   model_complexity=2).to(device)

    # Initialize an optimizer for the model parameters
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Initialize the HCQT feature extraction module
    hcqt = LHVQT(fs=sample_rate,
                 hop_length=hop_length,
                 fmin=librosa.midi_to_hz(fmin),
                 n_bins=n_bins,
                 bins_per_octave=bins_per_octave,
                 harmonics=harmonics,
                 update=False,
                 db_to_prob=False,
                 batch_norm=False).to(device)

    # Define the collection of augmentations to use for timbre invariance
    transforms = Compose(
        transforms=[
            # TODO - add more augmentations
            PolarityInversion(p=0.5)
        ]
    )

    # Instantiate Bach10 dataset for validation
    bach10 = Bach10(sample_rate=sample_rate,
                    hop_length=hop_length,
                    fmin=fmin,
                    n_bins=n_bins,
                    bins_per_octave=bins_per_octave,
                    seed=seed)

    # Initialize a list to hold all validation datasets
    validation_sets = [bach10]

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

            with torch.no_grad():
                # Feed the audio through the augmentation pipeline
                augmentations = transforms(audio, sample_rate=sample_rate)
                # Create random mixtures of the audio and keep track of mixing
                mixtures, legend = get_random_mixtures(audio)

            # TODO - mixed precision (amp/apex) for speedup?
            #with torch.autocast(device_type=f'cuda'):
            # Obtain spectral features
            original_features = decibels_to_linear(hcqt(audio))
            augment_features = decibels_to_linear(hcqt(augmentations))
            mixture_features = decibels_to_linear(hcqt(mixtures))

            # Compute pitch salience embeddings
            original_embeddings = model(original_features).squeeze()
            augment_embeddings = model(augment_features).squeeze()
            mixture_embeddings = model(mixture_features).squeeze()

            # Convert logits to activations (implicit pitch salience)
            original_salience = torch.sigmoid(original_embeddings)
            #augment_salience = torch.sigmoid(augment_embeddings)
            #mixture_salience = torch.sigmoid(mixture_embeddings)

            # TODO - some of the following losses can be applied to more than one (originals|augmentations|mixtures)

            # Compute the support loss with respect to the first harmonic for this batch
            support_loss = compute_support_loss(original_embeddings, original_features[:, 1])

            # Log the support loss for this batch
            writer.add_scalar('train/loss/support', support_loss, batch_count)

            # Compute the content loss for this batch
            content_loss = compute_content_loss(original_salience, original_features)

            # Log the content loss for this batch
            writer.add_scalar('train/loss/content', content_loss, batch_count)

            # Compute the linearity loss for this batch
            linearity_loss = compute_linearity_loss(mixture_embeddings, original_salience, legend)

            # Log the linearity loss for this batch
            writer.add_scalar('train/loss/linearity', linearity_loss, batch_count)

            # Compute the invariance loss for this batch
            invariance_loss = compute_contrastive_loss(original_embeddings.transpose(-1, -2),
                                                       augment_embeddings.transpose(-1, -2))

            # Log the invariance loss for this batch
            writer.add_scalar('train/loss/invariance', invariance_loss, batch_count)

            # Compute the translation loss for this batch
            translation_loss = compute_translation_loss(model, original_features, original_salience)

            # Log the translation loss for this batch
            writer.add_scalar('train/loss/translation', translation_loss, batch_count)

            # Compute the total loss for this batch
            loss = multipliers['support'] * support_loss + \
                   multipliers['content'] * content_loss + \
                   multipliers['linearity'] * linearity_loss + \
                   multipliers['invariance'] * invariance_loss + \
                   multipliers['translation'] * translation_loss

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
                for set in validation_sets:
                    # Validate the model with each validation dataset
                    evaluate(model=model,
                             hcqt=hcqt,
                             eval_set=set,
                             writer=writer,
                             i=batch_count,
                             device=device)

                # Place model back in training mode
                model.train()

                # Save the model checkpoint after each epoch is complete
                torch.save(model, os.path.join(log_dir, f'model-{batch_count}.pt'))
