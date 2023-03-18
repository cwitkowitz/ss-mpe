# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from objectives import *
from utils import seed_everything
from NSynth import NSynth
from model import SAUNet
from lhvqt import LHVQT

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

# Initialize the HCQT feature extraction module
# TODO - verify this is resembling librosa for some weird looking samples
hcqt = LHVQT(fs=16000,
             hop_length=512,
             n_bins=216,
             bins_per_octave=36,
             update=False,
             batch_norm=False).to(gpu_id)

# Initialize a PyTorch dataloader for the data
loader = DataLoader(dataset=nsynth,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=0,
                    drop_last=True)

n_channels = len(hcqt.harmonics)

# Initialize MPE representation learning model
model = SAUNet(n_ch_in=n_channels,
               n_bins_in=216,
               model_complexity=2).to(gpu_id)

# Initialize an optimizer for the model parameters
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

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
        audio = audio.to(gpu_id).float()

        # TODO - perform augmentations on audio here

        with torch.no_grad():
            # Create random mixtures of the audio and keep track of mixing
            mixtures, legend = get_random_mixtures(audio)

        # Add a channel dimension to the audio
        audio = audio.unsqueeze(-2)

        # Obtain spectral features for the audio
        features = hcqt(audio)

        """
        from librosa.display import specshow
        import matplotlib.pyplot as plt
        import librosa
        import numpy as np

        def cosine_similarity(a, b):
            assert len(a.shape) == 2
            assert a.shape == b.shape

            # Compute the dot product of matrix a and b as if they were vectors
            ab_dot = np.trace(np.dot(a.T, b))
            # Compute the norms of each matrix
            a_norm = np.linalg.norm(a)
            b_norm = np.linalg.norm(b)

            # Compute cosine similarity
            cos_sim = ab_dot / (a_norm * b_norm)

            return cos_sim

        test_librosa = librosa.cqt(audio[:, 0].cpu().detach().numpy(),
                                   sr=hcqt.tfs.lvqt2.fs,
                                   hop_length=hcqt.tfs.lvqt2.hop_length,
                                   fmin=hcqt.tfs.lvqt2.fmin,
                                   n_bins=hcqt.tfs.lvqt2.n_bins,
                                   bins_per_octave=hcqt.tfs.lvqt2.bins_per_octave)
        test_librosa = 1 + librosa.amplitude_to_db(test_librosa, ref=np.max) / 80

        fig, ((ax1, ax2)) = plt.subplots(2, 1)
        plt.sca(ax1)
        specshow(test_librosa[0],
                 sr=hcqt.tfs.lvqt2.fs,
                 hop_length=hcqt.tfs.lvqt2.hop_length,
                 fmin=hcqt.tfs.lvqt2.fmin,
                 bins_per_octave=hcqt.tfs.lvqt2.bins_per_octave,
                 x_axis='time',
                 y_axis='cqt_hz')
        ax1.set_title('Librosa HCQT')
        plt.sca(ax2)
        specshow(features[0, 1].cpu().detach().numpy(),
                 sr=hcqt.tfs.lvqt2.fs,
                 hop_length=hcqt.tfs.lvqt2.hop_length,
                 fmin=hcqt.tfs.lvqt2.fmin,
                 bins_per_octave=hcqt.tfs.lvqt2.bins_per_octave,
                 x_axis='time',
                 y_axis='cqt_hz')
        ax2.set_title(f'LHCQT')
        plt.show(block=True)
        """

        # Obtain an implicit salience map for the audio
        salience = model(features)

        # Convert the logits to activations
        activations = torch.sigmoid(salience)

        # Compute the reconstruction loss for this batch
        reconstruction_loss = compute_bce_reconstruction_loss(features[:, 1], activations[:, 0])

        # Log the reconstruction loss for this batch
        writer.add_scalar('train/loss/reconstruction', reconstruction_loss, batch_count)

        # Compute the content loss for this batch
        content_loss = compute_content_loss(features, activations)

        # Log the content loss for this batch
        writer.add_scalar('train/loss/content', content_loss, batch_count)

        # Add a channel dimension to the mixtures
        mixtures = mixtures.unsqueeze(-2)

        # Obtain spectral features for the mixtures
        mixture_features = hcqt(mixtures)

        # Obtain an implicit salience map for the mixtures
        mixture_salience = model(mixture_features)

        # Convert the logits to mixed activations
        mixture_activations = torch.sigmoid(mixture_salience)

        # Compute the linearity loss for this batch
        linearity_loss = compute_linearity_loss(activations, mixture_activations, legend)

        # Log the linearity loss for this batch
        writer.add_scalar('train/loss/linearity', linearity_loss, batch_count)

        """
        # TODO - don't think contrastve loss should be computed on mixtures

        # Compute the invariance loss for this batch
        invariance_loss = compute_timbre_invariance_loss(audio, model, invariance_transforms)

        # Log the invariance loss for this batch
        writer.add_scalar('train/loss/invariance', invariance_loss, batch_count)
        """

        # Compute the total loss for this batch
        loss = reconstruction_loss + 0 * content_loss + 0 * linearity_loss #+ invariance_loss

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
            writer.add_image('train/vis/cqt', features[0, 1 : 2].flip(-2), batch_count)
            writer.add_image('train/vis/salience', salience[0, 0:].flip(-2), batch_count)

    # Save the model checkpoint after each epoch is complete
    torch.save(model, os.path.join(log_dir, f'model-{i + 1}.pt'))
