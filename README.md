## Self-Supervised Multi-Pitch Estimation (SS-MPE)

First we recommend installing a fresh anaconda environment
```
conda create --name ss-mpe python=3.10
conda activate ss-mpe
```

Our project utilizes published code from two repositories, ```timbre-trap``` and ```lhvqt```.
However, we have modified these slightly, so we provide our local copies.
First, install their requirements, then install them manually:
```
pip install -r timbre-trap/requirements.txt
pip install -e timbre-trap/
```
```
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

Baseline results can be reproduced with ```baselines.py```, though note that the Timbre-Trap model has no mechanism to perform chunk-based processing, so it will likely crash unless your PC has lots of RAM.

The code will attempt to download most datasets automatically, however some datasets are not readily available online, and must be requested.

We also invite you to play around with the various other analysis and visualizastion scripts.

Thanks for reviewing!
If you have any questions, please to not hesitate to reach out!
