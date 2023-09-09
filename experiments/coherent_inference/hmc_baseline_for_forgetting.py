import argparse

import matplotlib.pyplot as plt
import numpyro
from jax import random

from experiments.src.data import *
import experiments.src.factory

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
    prog='HMCBaseline',
    description='HMC baseline for forgetting experiment; trained on all samples at once')
parser.add_argument('--factory', choices=['big', 'small'])
parser.add_argument('--perm', action="store_true")
parser.add_argument('--D_X', type=int)  # Feature expansion dimension
parser.add_argument('--legend', action="store_true")
args = parser.parse_args()

factory = getattr(experiments.src.factory, args.factory)

factory.D_X = args.D_X
pretrain_data = LinearData(intercept=0.5, beta=0.0, D_X=args.D_X)

shifted_data = LinearData(intercept=-0.5, beta=0.0, D_X=args.D_X)
if args.perm:
    np.random.seed(0)
    random_perm = np.random.choice(np.arange(50), size=50, replace=False)
    shifted_data = PermutedData(shifted_data, random_perm)

concat_data = ConcatData([pretrain_data, shifted_data])

bnn = factory.bnn()
hmc = factory.map_then_hmc(bnn, concat_data)

hmc.train(random.PRNGKey(0))
hmc.make_predictions(random.PRNGKey(1))

fig, ax = plt.subplots()
hmc.make_plots(fig=fig, ax=ax, legend=False, plot_samples=True)
if args.legend:
    ax.legend(frameon=True)
ax.set_ylim(-2.0, 2.0)
ax.set_title("Full HMC posterior on both datasets")
fig.savefig(f"figs/hmc_baseline_for_forgetting_{args.factory}_{args.D_X}{'_perm' if args.perm else ''}.pdf")
