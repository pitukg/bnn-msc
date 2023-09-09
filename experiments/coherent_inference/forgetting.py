import argparse

import matplotlib.pyplot as plt
import numpyro
from jax import random

from experiments.src.data import *
import experiments.src.factory
from experiments.src.experiment import SequentialExperimentBlock

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
    prog='forgetting',
    description='Pre-trained prior forgetting experiment')
parser.add_argument('--factory', choices=['big', 'small'])
parser.add_argument('--D_X', type=int)  # Feature expansion dimension
parser.add_argument('--block')  # For what algo to pretrain with
args = parser.parse_args()

factory = getattr(experiments.src.factory, args.factory)

fig, axs = plt.subplots(figsize=(18, 9), nrows=2, ncols=3, sharey="all")
axs = axs.ravel()

factory.D_X = args.D_X
pretrain_data = LinearData(intercept=0.5, beta=0.0, D_X=args.D_X)

np.random.seed(0)
random_perm = np.random.choice(np.arange(50), size=50, replace=False)
shifted_data = PermutedData(LinearData(intercept=-0.5, beta=0.0, D_X=args.D_X), random_perm)

# Take prefixes of shifted data to see effect of data size
retrain_sizes = [1, 2, 5, 25, 50]
retrain_datasets = [DataSlice(shifted_data, slice(size)) for size in retrain_sizes]

bnn = factory.bnn()

# Pretrain
if args.block == "mfvi":
    bnn.BETA = factory.BETA
Block = getattr(factory, args.block)
block_name = dict(mfvi="VI", map_then_diag_laplace="Laplace", map_then_swag="SWAG")[args.block]
pretrain_experiment: SequentialExperimentBlock = Block(bnn, pretrain_data)
pretrain_experiment.train(random.PRNGKey(0))
pretrain_experiment.make_predictions(random.PRNGKey(1))
pretrain_experiment.make_plots(fig=fig, ax=axs[0], plot_samples=True)
axs[0].set_title(f"Prior learnt via {block_name}")

pretrained_prior = pretrain_experiment.posterior
del pretrain_experiment
pretrained_bnn = bnn.with_prior(*pretrained_prior)
del bnn

pretrained_bnn.BETA = 1.0

for i, retrain_data in enumerate(retrain_datasets):
    ax = axs[1+i]
    retrain_data_size = retrain_data.train[1].shape[0]
    # # train VI on pretrained prior
    # vi_experiment_on_pretrained = BasicMeanFieldGaussianVIExperiment(
    #     pretrained_bnn, retrain_data, num_samples=VI_NUM_SAMPLES, max_iter=VI_MAX_ITER, lr_schedule=VI_LR_SCHEDULE)
    # vi_experiment_on_pretrained.train(random.PRNGKey(0))
    # vi_experiment_on_pretrained.make_predictions(random.PRNGKey(1))
    # # Make plot
    # fig, ax = plt.subplots()
    # vi_experiment_on_pretrained.make_plots(fig=fig, ax=ax)
    # ax.set_title(f"VI retrained on {retrain_data_size} points")
    # ax.set_ylim(-2.0, 2.0)
    # fig.tight_layout()
    # fig.savefig(f"figs/vi-retrained-on-{retrain_data_size}.png")
    # del vi_experiment_on_pretrained

    # train HMC on pretrained prior
    hmc_experiment_on_pretrained = factory.map_then_hmc(pretrained_bnn, retrain_data)
    hmc_experiment_on_pretrained.train(random.PRNGKey(0))
    hmc_experiment_on_pretrained.make_predictions(random.PRNGKey(1))
    # Make plot
    hmc_experiment_on_pretrained.make_plots(fig=fig, ax=ax, plot_samples=True)
    ax.plot(pretrain_data.train[0][:, 1], pretrain_data.train[1], "x", color="#50190A", alpha=0.7,
            label="Previously seen data points")
    ax.set_title(f"HMC retrained on {retrain_data_size} point{'s' if retrain_data_size > 1 else ''}")
    ax.set_ylim(-2.0, 2.0)
    if i == 3:
        # Plot legend
        leg = ax.legend()
        fig.legend(leg.legend_handles, [t.get_text() for t in leg.texts], loc="outside lower center", ncol=5)
        leg.remove()

fig.savefig(f"figs/forgetting_{args.block}_{args.factory}_{args.D_X}.pdf")
