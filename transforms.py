# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

from torch import nn

import torch


# TODO - is it necessary to inherit from nn.Module - do I even need to compute gradient?

class TranscriptionIrrelevantTransform(nn.Module):
    """
    A transformation that can be applied to a piece of audio
    without affecting its score, e.g. timbre augmentation.
    """

    def __init__(self):
        """
        TODO.
        """

        pass

    def transform_audio(self, audio):
        """
        TODO.
        """

        pass


class TranscriptionRelevantTransform(TranscriptionIrrelevantTransform):
    """
    A transformation that proportionally affects a piece of
    audio and its score, e.g. pitch shift or speed change.
    """

    def transform_score(self, score):
        """
        TODO.
        """

        pass
