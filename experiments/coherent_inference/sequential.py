import argparse
from typing import Callable

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpyro
from jax import random

from experiments.src.data import *
from experiments.src.experiment import SequentialExperimentBlock, BasicHMCExperiment
import experiments.src.factory
from experiments.src.model import BNNRegressor

DEVICE = "gpu"
numpyro.set_platform(DEVICE)

# Plotting
plt.rcParams.update({
    "axes.grid": True,  # show grid by default
    "axes.titlesize": 20,
    "axes.titlepad": 10.0,
    "axes.labelsize": 18,
    "figure.constrained_layout.use": True,
    "figure.titlesize": 22,
    "figure.subplot.wspace": 0.3,
    "font.size": 15,
    "font.weight": "normal",  # bold fonts
    "xtick.labelsize": 15,  # large tick labels
    "ytick.labelsize": 15,  # large tick labels
    "legend.frameon": False, # No frame on legend
    "lines.linewidth": 1,  # thick lines
    "lines.color": "k",  # black lines
    # "grid.color": "0.5",    # gray gridlines
    "grid.linestyle": "-",  # solid gridlines
    "grid.linewidth": 0.3,  # thin gridlines
    "savefig.dpi": 300,  # higher resolution output.
})


parser = argparse.ArgumentParser(
                    prog='SequentialUpdating',
                    description='Sequential updating experiment')
parser.add_argument('--factory', choices=['big', 'small'])
parser.add_argument('--D_X', type=int)
parser.add_argument('--block')
args = parser.parse_args()

factory = getattr(experiments.src.factory, args.factory)

# fig = plt.figure(figsize=(18, 9))
# gs = gridspec.GridSpec(2, 6)
#
# axs = []

fig, axs = plt.subplots(figsize=(18, 9), nrows=2, ncols=3, sharey='all')
axs = axs.ravel()

# for i in range(0, 5):
#     if i < 3:
#         if i == 0:
#             ax = plt.subplot(gs[0, 2 * i:2 * i + 2])
#         else:
#             ax = plt.subplot(gs[0, 2 * i:2 * i + 2], sharey=axs[0])
#             ax.sharey(axs[0])
#             plt.setp(ax.get_yticklabels(), visible=False)
#         ax.set_ylim(-4, 5)
#     else:
#         if i == 3:
#             ax = plt.subplot(gs[1, 2 * i - 5:2 * i + 2 - 5])
#         else:
#             ax = plt.subplot(gs[1, 2 * i - 5:2 * i + 2 - 5])
#             ax.sharey(axs[3])
#             plt.setp(ax.get_yticklabels(), visible=False)
#         ax.set_ylim(-2, 2.5)
#     axs.append(ax)

# Data & model
factory.D_X = args.D_X
data = ToyData1(feat_D_X=args.D_X, train_size=100)
bnn = factory.bnn()
if args.block == "mfvi":
    bnn.BETA = factory.BETA

# Training algorithms
Block: Callable[[BNNRegressor, Data], SequentialExperimentBlock] = getattr(factory, args.block)
NSTEPS = 5
curr, by = 0, 100 // NSTEPS
for i in range(NSTEPS):
    e = Block(bnn, DataSlice(data, slice(curr, curr + by)))
    e.train(random.PRNGKey(0))
    e.make_predictions(random.PRNGKey(1))
    e.make_plots(fig=fig, ax=axs[i], plot_samples=True)
    axs[i].plot(data.train[0][:curr, 1], data.train[1][:curr], "x", color="#50190A", alpha=0.7,
                label="Previously seen data points")
    axs[i].set_ylim(-6, 6)
    axs[i].set_title(f"Data points {curr}-{curr+by}")
    if i == 2:
        # Plot legend
        leg = axs[i].legend()
        fig.legend(leg.legend_handles, [t.get_text() for t in leg.texts], loc="outside lower center", ncol=5)
        leg.remove()

    if i+1 == NSTEPS:
        # Replace last experiment by HMC experiment too
        bnn.BETA = 1.0
        hmc: BasicHMCExperiment = factory.map_then_hmc(bnn, e._data)
        hmc.train(random.PRNGKey(0))
        hmc.make_predictions(random.PRNGKey(1))
        hmc.make_plots(fig=fig, ax=axs[-1], plot_samples=True)
        axs[-1].plot(data.train[0][:curr, 1], data.train[1][:curr], "x", color="#50190A", alpha=0.7)
        axs[-1].set_ylim(-6, 6)
        axs[-1].set_title(f"HMC on data points {curr}-{curr+by}")

    bnn = bnn.with_prior(e.posterior[0])
    curr += by

fig.savefig(f"figs/sequential_{args.block}_{args.factory}_{args.D_X}.pdf")
