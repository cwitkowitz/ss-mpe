# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from common import ComboSet
from FreeMusicArchive import FreeMusicArchive
from MagnaTagATune import MagnaTagATune
from NSynth import NSynth
from ToyNSynth import ToyNSynthTrain, ToyNSynthEval
from Bach10 import Bach10
from Su import Su
from TRIOS import TRIOS
from MusicNet import MusicNet
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


CONFIG = 1 # (0 - desktop | 1 - lab)
EX_NAME = '_'.join(['StepByStep'])

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
        'support' : 0.1,
        'content' : 1,
        'translation' : 1,
        'invariance' : 1,
        'linearity' : 0
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

    # Initialize the primary PyTorch device
    device = torch.device(f'cuda:{gpu_ids[0]}'
                          if torch.cuda.is_available() else 'cpu')

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

    # Determine the sequence length of training samples
    n_frames = int(n_secs * sample_rate / hop_length)

    # Initialize MPE representation learning model
    model = SAUNet(n_ch_in=len(harmonics),
                   n_bins_in=n_bins,
                   model_complexity=2,
                   #max_seq=4*n_frames)
                   )

    if len(gpu_ids) > 1:
        # Wrap feature extraction and model for multi-GPU usage
        hcqt = torch.nn.DataParallel(hcqt, device_ids=gpu_ids)
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)

    # Add model and feature extraction to primary device
    hcqt, model = hcqt.to(device), model.to(device)

    # Initialize an optimizer for the model parameters
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Choose cutoff frequencies to retain at least one octave
    low_cutoff = librosa.note_to_hz('C2')
    high_cutoff = librosa.note_to_hz('C6')

    # Define boundaries for musically-relevant fundamental frequencies
    low_bound = librosa.note_to_hz('A0')
    high_bound = librosa.note_to_hz('C8')

    def octave_fraction(n_octave):
        """
        Compute the fraction of an arbitrary frequency
        in Hz which corresponds to n_octave octaves.
        """

        return 2 ** (n_octave / 2) - 2 ** (-n_octave / 2)

    # Define a transformation pipeline to modify the timbre of audio
    timbre_transforms = Compose(
        transforms=[
            #AddColoredNoise(),
            #AddBackgroundNoise(),
            #ApplyImpulseResponse(),
            OneOf(
                transforms=[
                    LowPassFilter(
                        min_cutoff_freq=low_cutoff,
                        max_cutoff_freq=high_bound,
                        p=1.0
                    ),
                    HighPassFilter(
                        min_cutoff_freq=low_bound,
                        max_cutoff_freq=high_cutoff,
                        p=1.0
                    ),
                    BandPassFilter(
                        min_center_frequency=low_cutoff,
                        max_center_frequency=high_cutoff,
                        min_bandwidth_fraction=octave_fraction(1),
                        max_bandwidth_fraction=octave_fraction(2),
                        p=1.0
                    ),
                    BandStopFilter(
                        min_center_frequency=low_cutoff,
                        max_center_frequency=high_cutoff,
                        min_bandwidth_fraction=octave_fraction(1),
                        max_bandwidth_fraction=octave_fraction(2),
                        p=1.0
                    )
                ]
            )
        ]
    )

    # Define a transformation pipeline to add variance to audio
    variety_transforms = Compose(
        transforms=[
            # TODO - consider adding these augmentations
            #PolarityInversion(),
            #TimeInversion(),
            #PitchShift(),
            #Shift(),
            OneOf(
                transforms=[
                    #PeakNormalization(),
                    #Gain(),
                    Identity()
                ]
            )
            #ShuffleChannels()
        ]
    )

    # Connect the two types of transformations to obtain the full pipeline
    transforms = Compose(transforms=[timbre_transforms, variety_transforms])

    # Define maximum time and frequency shift
    max_shift_time = n_frames // 4
    max_shift_freq = 2 * bins_per_octave

    # Define time stretch boundaries
    min_stretch_time = 0.5
    max_stretch_time = 2

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

    # Instantiate Su dataset for validation
    toynsynthtest = ToyNSynthEval(base_dir=nsynth_base_dir,
                                  sample_rate=sample_rate,
                                  hop_length=hop_length,
                                  fmin=fmin,
                                  n_bins=n_bins,
                                  bins_per_octave=bins_per_octave)

    # Initialize a list to hold all validation datasets
    #validation_sets = [bach10, su, trios]
    validation_sets = [toynsynthtest]

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
            with torch.no_grad():
                # Feed the audio through the augmentation pipeline
                augmentations = transforms(audio, sample_rate=sample_rate)

                # Obtain another set of augmented embeddings for contrastive loss
                augmentations_ = transforms(audio, sample_rate=sample_rate)

                # Add all data to the appropriate device
                audio = audio.to(device)
                augmentations = augmentations.to(device)
                augmentations_ = augmentations_.to(device)

                # Create random mixtures of the audio and keep track of mixing
                #original_mixtures, original_legend = get_random_mixtures(audio)
                #augment_mixtures, augment_legend = get_random_mixtures(augmentations)

                # TODO - augment original mixtures for more examples per batch?

            with torch.autocast(device_type=f'cuda'):
                # Obtain spectral features
                original_features = hcqt(audio)
                augment_features = hcqt(augmentations)
                augment_features_ = hcqt(augmentations_)
                #mixture_o_features = hcqt(original_mixtures)
                #mixture_a_features = hcqt(augment_mixtures)

                original_features_l = decibels_to_amplitude(original_features)
                augment_features_l = decibels_to_amplitude(augment_features)
                augment_features_l_ = decibels_to_amplitude(augment_features_)
                #mixture_o_features_l = decibels_to_amplitude(mixture_o_features)
                #mixture_a_features_l = decibels_to_amplitude(mixture_a_features)

                original_features_s = rescale_decibels(original_features)
                augment_features_s = rescale_decibels(augment_features)
                augment_features_s_ = rescale_decibels(augment_features_)
                #mixture_o_features_s = rescale_decibels(mixture_o_features)
                #mixture_a_features_s = rescale_decibels(mixture_a_features)

                # Compute pitch salience embeddings
                original_embeddings = model(original_features_s).squeeze()
                augment_embeddings = model(augment_features_s).squeeze()
                augment_embeddings_ = model(augment_features_s_).squeeze()
                #mixture_o_embeddings = model(mixture_o_features_s).squeeze()
                #mixture_a_embeddings = model(mixture_a_features_s).squeeze()

                # Convert logits to activations (implicit pitch salience)
                original_salience = torch.sigmoid(original_embeddings)
                augment_salience = torch.sigmoid(augment_embeddings)
                #mixture_o_salience = torch.sigmoid(mixture_o_embeddings)
                #mixture_a_salience = torch.sigmoid(mixture_a_embeddings)

                # Compute the support loss with respect to the first harmonic for this batch
                support_loss = compute_support_loss(original_embeddings, original_features_l)
                support_loss += compute_support_loss(augment_embeddings, original_features_l)
                #support_loss += compute_support_loss(mixture_o_embeddings, original_features_s)
                #support_loss += compute_support_loss(mixture_a_embeddings, original_features_s)

                # Log the support loss for this batch
                writer.add_scalar('train/loss/support', support_loss, batch_count)

                # Compute the content loss for this batch
                content_loss = compute_content_loss(original_salience, original_features_l)
                content_loss += compute_content_loss(augment_salience, original_features_l)
                #content_loss += compute_content_loss(mixture_o_salience, original_features_s)
                #content_loss += compute_content_loss(mixture_a_salience, original_features_s)

                # Log the content loss for this batch
                writer.add_scalar('train/loss/content', content_loss, batch_count)

                # Compute the translation loss for this batch
                translation_loss = compute_translation_loss(model, original_features_s, original_salience,
                                                            max_shift_f=max_shift_freq, max_shift_t=max_shift_time,
                                                            min_stretch=min_stretch_time, max_stretch=max_stretch_time)
                translation_loss += compute_translation_loss(model, augment_features_s, augment_salience,
                                                             max_shift_f=max_shift_freq, max_shift_t=max_shift_time,
                                                             min_stretch=min_stretch_time, max_stretch=max_stretch_time)
                #translation_loss += compute_translation_loss(model, mixture_o_features_s, mixture_o_salience,
                #                                             max_shift_f=max_shift_freq, max_shift_t=max_shift_time,
                #                                             min_stretch=min_stretch_time, max_stretch=max_stretch_time)
                #translation_loss += compute_translation_loss(model, mixture_a_features_s, mixture_a_salience,
                #                                             max_shift_f=max_shift_freq, max_shift_t=max_shift_time,
                #                                             min_stretch=min_stretch_time, max_stretch=max_stretch_time)

                # Log the translation loss for this batch
                writer.add_scalar('train/loss/translation', translation_loss, batch_count)

                # Compute the invariance loss for this batch
                invariance_loss = compute_contrastive_loss(original_embeddings.transpose(-1, -2),
                                                           augment_embeddings.transpose(-1, -2)) / batch_size
                invariance_loss += compute_contrastive_loss(original_embeddings.transpose(-1, -2),
                                                            augment_embeddings_.transpose(-1, -2)) / batch_size
                invariance_loss += compute_contrastive_loss(augment_embeddings.transpose(-1, -2),
                                                            augment_embeddings_.transpose(-1, -2)) / batch_size

                # Log the invariance loss for this batch
                writer.add_scalar('train/loss/invariance', invariance_loss, batch_count)

                # Compute the linearity loss for this batch
                #linearity_loss = compute_linearity_loss(mixture_o_embeddings, original_salience, original_legend)
                #linearity_loss += compute_linearity_loss(mixture_a_embeddings, augment_salience, augment_legend)

                # Log the linearity loss for this batch
                #writer.add_scalar('train/loss/linearity', linearity_loss, batch_count)

                # Compute the total loss for this batch
                loss = multipliers['support'] / 2 * support_loss + \
                       multipliers['content'] / 2 * content_loss + \
                       multipliers['translation'] / 2 * translation_loss + \
                       multipliers['invariance'] * invariance_loss
                #loss = multipliers['support'] / 4 * support_loss + \
                       #multipliers['content'] / 4 * content_loss + \
                       #multipliers['translation'] / 4 * translation_loss + \
                       #multipliers['invariance'] * invariance_loss + \
                       #multipliers['linearity'] / 2 * linearity_loss

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
                    evaluate(model=model,
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
