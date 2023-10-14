# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from . import constants

# Regular imports
from torch.utils.data import default_collate


__all__ = [
    'collate_audio_only'
]


def collate_audio_only(batch):
    """
    Collate only audio data for a batch.

    Parameters
    ----------
    batch : list of dicts containing
      track : string
        Identifier for the track
      audio : Tensor (1 x N)
        Sampled audio for the track
      ...

    Returns
    ----------
    batch : None or dict containing
      tracks : list of string
        Identifiers for the batched tracks
      audio : Tensor (B x 1 x N)
        Sampled audio for batched tracks
    """

    for data in batch:
        # Loop through data entries
        for key in list(data.keys()):
            # Check if entry is track name or audio
            if not (key == constants.KEY_TRACK or
                    key == constants.KEY_AUDIO):
                # Remove entry
                data.pop(key)

    # Collate remaining data using standard procedure
    batch = default_collate(batch)

    return batch
