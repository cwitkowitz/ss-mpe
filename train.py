# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from objectives import compute_content_loss, compute_linearity_loss, compute_invariance_loss
from utils import seed_everything
from model import PlaceHolderNet
from NSynth import NSynth

# Regular imports
from torch.utils.tensorboard import SummaryWriter
from sacred.observers import FileStorageObserver
from torch.utils.data import DataLoader
from torch_audiomentations import *
from sacred import Experiment
from tqdm import tqdm

import torch
import os


EX_NAME = '_'.join(['Self-Supervised-Sandbox_invariance'])

ex = Experiment('Train a model to learn representations for MPE.')

# TODO - leverage amt-tools or keep as standalone?
# TODO - try using apex for faster training?


#@ex.config
#def config():
# Maximum number of training iterations to conduct
max_epochs = 1000

# Number of samples to gather for a batch
batch_size = 8

# Fixed learning rate
learning_rate = 1e-4

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

# Initialize MPE representation learning model
model = PlaceHolderNet().to(gpu_id)

# Initialize an optimizer for the model parameters
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Define the collection of augmentations to use when training for invariance
invariance_transforms = Compose(
    transforms=[
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
        audio = audio.to(gpu_id)

        # Compute the content loss for this batch
        content_loss = compute_content_loss(audio, model)

        # Log the content loss for this batch
        writer.add_scalar('train/loss/content', content_loss, batch_count)

        # Compute the linearity loss for this batch
        linearity_loss = compute_linearity_loss(audio, model)

        # Log the linearity loss for this batch
        writer.add_scalar('train/loss/linearity', linearity_loss, batch_count)

        # Compute the invariance loss for this batch
        invariance_loss = compute_invariance_loss(audio, model, invariance_transforms)

        # Log the invariance loss for this batch
        writer.add_scalar('train/loss/invariance', invariance_loss, batch_count)

        # Compute the total loss for this batch
        loss = linearity_loss + 10 * content_loss + invariance_loss

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

    # Save the model checkpoint after each epoch is complete
    torch.save(model, os.path.join(log_dir, f'model-{i + 1}.pt'))
