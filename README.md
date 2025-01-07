## Self-Supervised Multi-Pitch Estimation (SS-MPE)
Code for the paper "[Toward Fully Self-Supervised Multi-Pitch Estimation](https://arxiv.org/pdf/2402.15569)".

## Installation
Clone the following repositories and install them along with their requirements:

```
git clone -b updates https://github.com/sony/timbre-trap
pip install -r timbre-trap/requirements.txt
pip install -e timbre-trap/
```
```
git clone -b refresh https://github.com/cwitkowitz/lhvqt
pip install -r lhvqt/requirements.txt
pip install -e lhvqt/
```

Then, install the main package ```ss-mpe```:
```
pip install -r ss-mpe/requirements.txt
pip install -e ss-mpe/
```

All code for experiments and visualization is located under ```ss-mpe/experiments```.
To reproduce our experiments, simply run ```train.py``` and update the ```multipliers``` parameter to reflect the desired loss configuration
You may also want to update ```EX_NAME``` and ```root_dir``` to your liking. 

To evaluate an existing model, run ```comparisons.py``` with the model and checkpoint selected.
Again, make sure all the paths are set correctly / to your liking.
Baseline results can be reproduced with ```baselines.py```, however note that there may be issues with attempting to run the script within a CUDA environment.
