# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from NSynth import NSynth
from model import SAUNet
from lhvqt import LHVQT
from objectives import *
from utils import *

# Regular imports
from torch.utils.tensorboard import SummaryWriter
from sacred.observers import FileStorageObserver
from torch.utils.data import DataLoader
from torch_audiomentations import *
from sacred import Experiment
from tqdm import tqdm

import torch
import os


EX_NAME = '_'.join(['Sandbox'])

ex = Experiment('Train a model to learn representations for MPE.')

# TODO - leverage amt-tools or keep as standalone?
# TODO - try using apex for faster training?


#@ex.config
#def config():
# Maximum number of training iterations to conduct
max_epochs = 1000

# Number of iterations between checkpoints
checkpoint_interval = 50

# Number of samples to gather for a batch
batch_size = 32

# Fixed learning rate
learning_rate = 1e-3

# ID of the gpu to use, if available
gpu_id = 0

# Random seed for this experiment
seed = 0

# Create the root directory for the experiment files
root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'generated', 'experiments', EX_NAME)
os.makedirs(root_dir, exist_ok=True)

# Add a file storage observer for the log directory
ex.observers.append(FileStorageObserver(root_dir))


#@ex.automain
#def learn_representations(max_epochs, batch_size, learning_rate, gpu_id, seed, root_dir):
# Seed everything with the same seed
seed_everything(seed)

# Construct the path pointing to the NSynth dataset
dataset_base_dir = os.path.join(os.path.expanduser('~'), 'Desktop', 'Datasets', 'NSynth')

# Instantiate a view on the NSynth data
nsynth = NSynth(base_dir=dataset_base_dir,
                seed=seed)

# Initialize a PyTorch dataloader for the data
loader = DataLoader(dataset=nsynth,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=0,
                    drop_last=True)

# Define some input parameters
n_bins = 216
sample_rate = 16000
harmonics = [0.5, 1, 2, 3, 4, 5]

# Initialize MPE representation learning model
model = SAUNet(n_ch_in=len(harmonics),
               n_bins_in=n_bins,
               model_complexity=1).to(gpu_id)

# Initialize an optimizer for the model parameters
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Initialize the HCQT feature extraction module
# TODO - need to make sure features for silence or near-silence are very tiny
hcqt = LHVQT(fs=sample_rate,
             hop_length=512,
             n_bins=n_bins,
             bins_per_octave=36,
             harmonics=harmonics,
             db_to_prob=False,
             update=False,
             batch_norm=False).to(gpu_id)

# Define the collection of augmentations to use for timbre invariance
transforms = Compose(
    transforms=[
        # TODO - add more augmentations
        PolarityInversion(p=0.5)
    ]
)

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
        # Add the audio to the appropriate GPU
        audio = audio.to(gpu_id).float()

        # Add a channel dimension to the audio
        audio = audio.unsqueeze(-2)

        with torch.no_grad():
            # Feed the audio through the augmentation pipeline
            augmentations = transforms(audio, sample_rate=sample_rate)
            # Create random mixtures of the audio and keep track of mixing
            mixtures, legend = get_random_mixtures(audio)

        #with torch.autocast(device_type=f'cuda'):
        # Obtain spectral features
        original_features = decibels_to_linear(hcqt(audio))
        #augment_features = decibels_to_linear(hcqt(augmentations))
        mixture_features = decibels_to_linear(hcqt(mixtures))

        # Compute pitch salience embeddings
        original_embeddings = model(original_features).squeeze()
        #augment_embeddings = model(augment_features).squeeze()
        mixture_embeddings = model(mixture_features).squeeze()

        # Convert logits to activations (implicit pitch salience)
        original_salience = torch.sigmoid(original_embeddings)
        #augment_salience = torch.sigmoid(augment_embeddings)
        #mixture_salience = torch.sigmoid(mixture_embeddings)

        # TODO - some of the following losses can be applied to more than one (originals|augmentations|mixtures)

        # Compute the reconstruction loss with respect to the first harmonic for this batch
        reconstruction_loss = compute_reconstruction_loss(original_embeddings, original_features[:, 1])

        # Log the reconstruction loss for this batch
        writer.add_scalar('train/loss/reconstruction', reconstruction_loss, batch_count)

        # Compute the content loss for this batch
        content_loss = compute_content_loss(original_features, original_salience)

        # Log the content loss for this batch
        writer.add_scalar('train/loss/content', content_loss, batch_count)

        # Compute the linearity loss for this batch
        linearity_loss = compute_linearity_loss(mixture_embeddings, original_salience, legend)

        # Log the linearity loss for this batch
        writer.add_scalar('train/loss/linearity', linearity_loss, batch_count)

        """
        # Compute the invariance loss for this batch
        invariance_loss = compute_contrastive_loss(original_embeddings, augment_embeddings)

        # Log the invariance loss for this batch
        writer.add_scalar('train/loss/invariance', invariance_loss, batch_count)

        # Compute the translation loss for this batch
        translation_loss = compute_translation_loss(model, original_features, original_salience)

        # Log the translation loss for this batch
        writer.add_scalar('train/loss/translation', translation_loss, batch_count)
        """

        # Compute the total loss for this batch
        loss = 0 * reconstruction_loss + 1 * content_loss + 1 * linearity_loss #+ 0 * invariance_loss + 0 * translation_loss

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
            # Log the input features and output salience for this batch
            writer.add_image('train/vis/cqt', original_features[0, 1 : 2].flip(-2), batch_count)
            writer.add_image('train/vis/salience', original_salience[0].unsqueeze(0).flip(-2), batch_count)

    # Save the model checkpoint after each epoch is complete
    torch.save(model, os.path.join(log_dir, f'model-{i + 1}.pt'))
