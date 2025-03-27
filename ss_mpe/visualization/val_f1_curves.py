# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# Regular imports
from tensorflow.python.summary.summary_iterator import summary_iterator
from collections import defaultdict

import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import os


# Experiments to plot
experiments = {
    'base/URMP_SPV_T_G_P_LR5E-4_2_BS8_MC3_W100_TTFC' : ('Ref.', 'black'),
    #'additional/URMP_SPV_T_G_P_+NSynth_LR5E-4_2_BS24_R0.66_MC3_W100_TTFC' : ('+NSynth', 'gold'),
    #'energy/URMP_SPV_T_G_P_+NSynth_EG_SPR_LR5E-4_2_BS24_R0.66_MC3_W100_TTFC' : ('+NSynth+EG', 'forestgreen'),
    #'additional/URMP_SPV_T_G_P_+MNet_LR5E-4_2_BS24_R0.66_MC3_W100_TTFC' : ('+MusicNet', 'red'),
    #'two-stage/URMP_SPV_T_G_P_-_+MNet_LR1E-4_2_BS24_R0.66_MC3_W100_TTFC' : ('+MusicNet-2S', 'orangered'),
    #'energy/URMP_SPV_T_G_P_+MNet_EG_SPR_LR5E-4_2_BS24_R0.66_MC3_W100_TTFC' : ('+MusicNet+EG', 'darkorange'),
    #'additional/URMP_SPV_T_G_P_+FMA_LR5E-4_2_BS24_R0.66_MC3_W100_TTFC' : ('+FMA', 'darkcyan'),
    #'two-stage/URMP_SPV_T_G_P_-_+FMA_LR1E-4_2_BS24_R0.66_MC3_W100_TTFC' : ('+FMA-2S', 'blue'),
    #'energy/URMP_SPV_T_G_P_+FMA_EG_SPR_LR5E-4_2_BS24_R0.66_MC3_W100_TTFC' : ('+FMA+EG', 'purple'),

    #'additional/URMP_SPV_T_G_P_+MNet_LR5E-4_2_BS10_R0.2_MC3_W100_TTFC' : ('+MN-2', 'gold'),
    #'additional/URMP_SPV_T_G_P_+MNet_LR5E-4_2_BS12_R0.33_MC3_W100_TTFC' : ('+MN-4', 'gold'),
    #'additional/URMP_SPV_T_G_P_+MNet_LR5E-4_2_BS16_R0.5_MC3_W100_TTFC' : ('+MN-8', 'darkorange'),
    'additional/URMP_SPV_T_G_P_+MNet_LR5E-4_2_BS24_R0.66_MC3_W100_TTFC' : ('+MN-16', 'red'),
    #'energy/URMP_SPV_T_G_P_+MNet_EG_SPR_LR5E-4_2_BS24_R0.66_MC3_W100_TTFC' : ('+MN+EG', 'arkorange'),
    #'two-stage/URMP_SPV_T_G_P_-_+MNet_LR1E-4_2_BS24_R0.66_MC3_W100_TTFC' : ('+MN-16-FT', 'brown')

    #'dist/URMP-T1_SPV_T_G_P_LR5E-4_2_BS8_MC3_W100_TTFC' : ('T1', 'purple'),
    #'dist/URMP-T1_SPV_T_G_P_+URMP-T2_LR5E-4_2_BS18_R0.56_MC3_W100_TTFC' : ('T1/T2', 'blue'),
    #'dist/URMP-T1_SPV_T_G_P_+URMP-T2_EG_SPR_LR5E-4_2_BS18_R0.56_MC3_W100_TTFC' : ('T1/T2+EG', 'green')
}

# File layout of system (0 - desktop | 1 - lab)
path_layout = 0

# Construct the path to the top-level directory of the experiments
if path_layout == 1:
    experiment_dir = os.path.join('/', 'storage', 'frank', 'self-supervised-pitch')
else:
    #experiment_dir = os.path.join('..', 'generated', 'experiments', ex_name)
    experiment_dir = f'/media/rockstar/Icarus/ss-mpe_ISMIR_overfitting_degeneration/'

idcs = {
    'URMP' : (0, 0),
    'Bach10' : (0, 1),
    'Su' : (0, 2),
    'TRIOS' : (1, 0),
    'MusicNet' : (1, 1),
    'GuitarSet' : (1, 2)
}

plt.rcParams.update({'axes.titlesize': 'x-large'})
plt.rcParams.update({'axes.labelsize': 'large'})
plt.rcParams.update({'xtick.labelsize': 'large'})
plt.rcParams.update({'ytick.labelsize': 'large'})

#plt.rcParams.update({'legend.fontsize': 'large'})
plt.rcParams.update({'legend.fontsize': 'x-large'})
#plt.rcParams.update({'figure.titlesize': 1.25 * plt.rcParams['figure.titlesize']})
#plt.rcParams.update({'grid.linewidth': 1.25 * plt.rcParams['grid.linewidth']})
plt.rcParams.update({'lines.linewidth': 1.25 * plt.rcParams['lines.linewidth']})

#plt.rcParams.update({'font.size': 1.25 * plt.rcParams['font.size']})

labels = [v[0] for v in experiments.values()]

# Open a new figure
fig, axes = plt.subplots(2, 3, figsize=(18, 7.5))
axes[idcs['URMP']].set_title('URMP')
axes[idcs['Bach10']].set_title('Bach10')
axes[idcs['Su']].set_title('Su')
axes[idcs['TRIOS']].set_title('TRIOS')
axes[idcs['MusicNet']].set_title('MusicNet')
axes[idcs['GuitarSet']].set_title('GuitarSet')

for exp, (tag, color) in experiments.items():
    # Construct the path to the experiment checkpoints
    models_dir = os.path.join(experiment_dir, exp, 'models')
    # Obtain the events file associated with the experiment
    events_files = [f for f in os.listdir(models_dir) if f.startswith('events.out.tfevents')]
    # Choose the events file with the largest memory footprint
    events_file = events_files[np.argmax([os.path.getsize(os.path.join(models_dir, e)) for e in events_files])]
    # Construct the path to the events file
    events_path = os.path.join(models_dir, events_file)

    #https://github.com/wookayin/expt/blob/v0.3.0/expt/data.py#L831-L903
    def iter_scalar_summary_from_event_file(event_file):
        try:
            for event in summary_iterator(event_file):
                step = int(event.step)
                if not event.HasField('summary'):
                    continue
                for value in event.summary.value:
                    if value.HasField('simple_value'):  # v1
                        node = value
                        for p in 'simple_value'.split('.'):
                            node = getattr(node, p, None)
                            if node is None:
                                node = None
                                break
                        simple_value = node
                        yield step, value.tag, simple_value
        except tf.errors.DataLossError:
            print('Encountered a truncated record, skipping the rest.')


    # int(timestep) -> dict of columns
    all_data = defaultdict(dict)  # type: ignore
    for step, tag_name, value in iter_scalar_summary_from_event_file(events_path):
        all_data[step][tag_name] = value

    for t in list(all_data.keys()):
        all_data[t]['global_step'] = t

    df = pd.DataFrame(all_data).T
    df = df[:10000].dropna()

    steps = df.index.to_numpy()
    f1_urmp_val = df['URMP/mpe/f1-score'].to_numpy()
    f1_bach10 = df['Bach10/mpe/f1-score'].to_numpy()
    f1_su = df['Su/mpe/f1-score'].to_numpy()
    f1_trios = df['TRIOS/mpe/f1-score'].to_numpy()
    f1_mnet_test = df['MusicNet/mpe/f1-score'].to_numpy()
    f1_gset_val = df['GuitarSet/mpe/f1-score'].to_numpy()

    axes[idcs['URMP']].plot(steps, f1_urmp_val, label=tag, color=color, alpha=0.8)
    axes[idcs['Bach10']].plot(steps, f1_bach10, color=color, alpha=0.8)
    axes[idcs['Su']].plot(steps, f1_su, color=color, alpha=0.8)
    axes[idcs['TRIOS']].plot(steps, f1_trios, color=color, alpha=0.8)
    axes[idcs['MusicNet']].plot(steps, f1_mnet_test, color=color, alpha=0.8)
    axes[idcs['GuitarSet']].plot(steps, f1_gset_val, color=color, alpha=0.8)

for (i, j) in idcs.values():
    axes[i, j].set_ylim(-0.05, 1.05)
    axes[i, j].set_yticks(np.arange(21) * 0.05, minor=True)
    axes[i, j].grid(which='both')
    axes[i, j].set_xlabel('# Batches')
    axes[i, j].set_ylabel('$\mathit{F_1}$-Score')
# Add a legend below the subplots
#fig.legend(labels, loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=9, frameon=False)
#fig.legend(labels, loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False)
fig.legend(labels, loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False)

# Open the figure manually
plt.show(block=False)

#plt.subplots_adjust(bottom=0.2, hspace=0.4, wspace=0.3)
fig.tight_layout(rect=[0, 0.05, 1, 1])

# Wait for keyboard input
while plt.waitforbuttonpress() != True:
    continue

# Prompt user to save figure
save = input('Save figure? (y/n)')

if save == 'y':
    # Create a directory for saving visualized loss curves
    save_dir = os.path.join('..', '..', 'generated', 'visualization')
    # Construct save path under visualization directory
    save_path = os.path.join(save_dir, f'f1_curves.pdf')
    # Save the figure with minimal whitespace
    fig.savefig(save_path, bbox_inches='tight', pad_inches=0)

# Close figure
plt.close(fig)
