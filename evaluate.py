# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from utils import *

# Regular imports
from torch.utils.data import DataLoader


def evaluate(model, hcqt, eval_set, writer=None):
    # Initialize a PyTorch dataloader for the data
    loader = DataLoader(dataset=eval_set,
                        batch_size=1,
                        shuffle=False,
                        num_workers=0,
                        drop_last=False)

    with torch.no_grad():
        # Loop through each testing track
        for audio, ground_truth in loader:
            # Obtain features for the audio
            features = decibels_to_linear(hcqt(audio))
            # Compute the pitch salience of the features
            # TODO - need to train with longer sizes
            salience = torch.sigmoid(model(features).squeeze())

            # TODO - evaluate pitch salience (https://github.com/cwitkowitz/amt-tools/blob/b41ead77a348157caaeec57243f30be8f5536330/amt_tools/evaluate.py#L794)

            # TODO - log results to writer
