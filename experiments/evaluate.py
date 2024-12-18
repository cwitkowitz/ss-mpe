# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from timbre_trap.datasets import NoteDataset
from ss_mpe.objectives import *
from timbre_trap.utils import *

# Regular imports
import librosa
import torch


def evaluate(model, eval_set, multipliers, writer=None, i=0, device='cpu', eq_kwargs=None, gm_kwargs=None, pc_kwargs=None, an_kwargs=None, dp_kwargs=None):
    # Initialize a new evaluator for the dataset
    evaluator = MultipitchEvaluator()

    # Add model to selected device and switch to evaluation mode
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        # Loop through tracks
        for k in range(eval_set.__len__()):
            # Extract the track data (always from t=0)
            data = eval_set.__getitem__(k, offset_s=0)
            # Determine which track is being processed
            track = data[constants.KEY_TRACK]
            # Extract audio and add to the appropriate device
            audio = data[constants.KEY_AUDIO].to(device).unsqueeze(0)
            # Extract ground-truth targets as a Tensor and add to the appropriate device
            ground_truth = torch.Tensor(data[constants.KEY_GROUND_TRUTH]).to(device).unsqueeze(0)

            if isinstance(eval_set, NoteDataset):
                # Extract frame times of ground-truth targets as reference
                times_ref = data[constants.KEY_TIMES]
                # Obtain the ground-truth note annotations
                pitches, intervals = eval_set.get_ground_truth(track)
                # Convert note pitches to Hertz
                pitches = librosa.midi_to_hz(pitches)
                # Convert the note annotations to multi-pitch annotations
                multi_pitch_ref = eval_set.notes_to_multi_pitch(pitches, intervals, times_ref)
            else:
                # Obtain the ground-truth multi-pitch annotations
                times_ref, multi_pitch_ref = eval_set.get_ground_truth(track)

                if eval_set.n_secs is not None:
                    # Slice times and multi-pitch annotations
                    times_ref = times_ref[times_ref <= eval_set.n_secs]
                    multi_pitch_ref = multi_pitch_ref[:len(times_ref)]

            # Compute full set of spectral features
            features = model.get_all_features(audio)

            # Extract relevant feature sets
            features_db   = features['db']
            features_db_1 = features['db_1']
            features_db_h = features['db_h']

            # Process features to obtain logits
            logits = model(features_db)
            # Convert to (implicit) pitch salience activations
            raw_activations = torch.sigmoid(logits)

            # Determine the times associated with predictions
            times_est = model.hcqt.get_times(model.hcqt.get_expected_frames(audio.size(-1)))
            # Perform peak-picking and thresholding on the activations
            activations = threshold(filter_non_peaks(to_array(raw_activations)), 0.5).squeeze(0)

            # Convert the activations to frame-level multi-pitch estimates
            multi_pitch_est = eval_set.activations_to_multi_pitch(activations, model.hcqt.get_midi_freqs())

            # Compute results for this track using mir_eval multi-pitch metrics
            results = evaluator.evaluate(times_est, multi_pitch_est, times_ref, multi_pitch_ref)
            # Store the computed results
            evaluator.append_results(results)

            # TODO - the following is super similar to training loop

            # Compute energy loss w.r.t. weighted harmonic sum for the track
            energy_loss = compute_energy_loss(logits, features_db_h)
            # Store the energy loss for the track
            evaluator.append_results({'loss/energy' : energy_loss.item()})

            # Compute support loss w.r.t. first harmonic for the track
            support_loss = compute_support_loss(logits, features_db_1)
            # Store the support loss for the track
            evaluator.append_results({'loss/support' : support_loss.item()})

            # Compute harmonic loss w.r.t. weighted harmonic sum for the track
            harmonic_loss = compute_harmonic_loss(logits, features_db_h)
            # Store the harmonic loss for the track
            evaluator.append_results({'loss/harmonic' : harmonic_loss.item()})

            # Compute sparsity loss for the track
            sparsity_loss = compute_sparsity_loss(raw_activations)
            # Store the sparsity loss for the track
            evaluator.append_results({'loss/sparsity' : sparsity_loss.item()})

            # Compute entropy loss for the track
            entropy_loss = compute_entropy_loss(logits)
            # Store the entropy loss for the track
            evaluator.append_results({'loss/entropy' : entropy_loss.item()})

            # Compute supervised BCE loss for the batch
            supervised_loss = compute_supervised_loss(logits, ground_truth)
            # Store the supervised loss for the track
            evaluator.append_results({'loss/supervised' : supervised_loss.item()})

            # Compute the total loss for the track
            total_loss = multipliers['energy'] * energy_loss + \
                         multipliers['support'] * support_loss + \
                         multipliers['harmonic'] * harmonic_loss + \
                         multipliers['sparsity'] * sparsity_loss + \
                         multipliers['entropy'] * entropy_loss + \
                         multipliers['supervised'] * supervised_loss

            # Compute feature-invariance loss for the track
            feature_loss = compute_feature_loss(model, features_db, raw_activations, **dp_kwargs)
            # Store the feature-invariance loss for the track
            # TODO - update to feature when convenient
            evaluator.append_results({'loss/channel' : feature_loss.item()})
            # Add the feature-invariance loss to the total loss
            total_loss += multipliers['feature'] * feature_loss

            if eq_kwargs is not None:
                # Compute timbre-invariance loss for the track using specified equalization
                timbre_loss = compute_timbre_loss(model, features_db, raw_activations, **eq_kwargs)
                # Store the timbre-invariance loss for the track
                evaluator.append_results({'loss/timbre' : timbre_loss.item()})
                # Add the timbre-invariance loss to the total loss
                total_loss += multipliers['timbre'] * timbre_loss

            if gm_kwargs is not None:
                # Compute geometric-equivariance loss for the track
                geometric_loss = compute_geometric_loss(model, features_db, raw_activations, **gm_kwargs)
                # Store the geometric-equivariance loss for the track
                evaluator.append_results({'loss/geometric' : geometric_loss.item()})
                # Add the geometric-equivariance loss to the total loss
                total_loss += multipliers['geometric'] * geometric_loss

            if pc_kwargs is not None:
                # Compute percussion-invariance loss for the batch
                percussion_loss = compute_percussion_loss(model, audio, raw_activations, **pc_kwargs)
                # Store the percussion-invariance loss for the track
                evaluator.append_results({'loss/percussion' : percussion_loss.item()})
                # Add the percussion-invariance loss to the total loss
                total_loss += multipliers['percussion'] * percussion_loss

            if an_kwargs is not None:
                # Compute noise-invariance loss for the batch
                noise_loss = compute_noise_loss(model, audio, raw_activations, **an_kwargs)
                # Store the noise-invariance loss for the track
                evaluator.append_results({'loss/noise' : noise_loss.item()})
                # Add the noise-invariance loss to the total loss
                total_loss += multipliers['noise'] * noise_loss

            # Store the total loss for the track
            evaluator.append_results({'loss/total' : total_loss.item()})

        # Compute the average for all scores
        average_results, _ = evaluator.average_results()

        if writer is not None:
            # Loop through all computed scores
            for key in average_results.keys():
                # Log the average score for this dataset
                writer.add_scalar(f'{eval_set.name()}/{key}', average_results[key], i)

            # Add channel dimension to input/outputs
            features_log_1 = features_db_1.unsqueeze(-3)
            features_log_h = features_db_h.unsqueeze(-3)
            transcription = raw_activations.unsqueeze(-3)
            ground_truth = ground_truth.unsqueeze(-3)

            # Remove batch dimension from inputs
            features_log_1 = features_log_1.squeeze(0)
            features_log_h = features_log_h.squeeze(0)
            transcription = transcription.squeeze(0)
            ground_truth = ground_truth.squeeze(0)

            # TODO - plot constants images only for first validation loop?

            # Visualize predictions for the final sample of the evaluation dataset
            writer.add_image(f'{eval_set.name()}/CQT (dB)', features_log_1.flip(-2), i)
            writer.add_image(f'{eval_set.name()}/W.Avg. HCQT', features_log_h.flip(-2), i)
            writer.add_image(f'{eval_set.name()}/transcription', transcription.flip(-2), i)
            writer.add_image(f'{eval_set.name()}/ground-truth', ground_truth.flip(-2), i)

    return average_results
