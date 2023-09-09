import argparse

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpyro
from jax import random

import experiments.src.factory
from experiments.src.data import *
from experiments.src.experiment import *

DEVICE = "gpu"
numpyro.set_platform(DEVICE)

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
    prog='ComplexityComparison',
    description='Comparison of different algorithms')
parser.add_argument('--factory', choices=['big', 'small'])
parser.add_argument('--D_X', type=int)
parser.add_argument('--data')
parser.add_argument('--bald', action='store_true')
args = parser.parse_args()

factory: experiments.src.factory.factory = getattr(experiments.src.factory, args.factory)
factory.D_X = args.D_X
if args.data == "toy":
    data = ToyData1(feat_D_X=factory.D_X, train_size=100)
elif args.data == "half_toy":
    data = DataSlice(ToyData1(feat_D_X=factory.D_X, train_size=100), slice(0, 50))
elif args.data == "flat":
    data = DataSlice(ToyData1(feat_D_X=factory.D_X, gen_D_X=2, train_size=100), slice(0, 50))
elif args.data == "linear":
    data = DataSlice(LinearData(intercept=1, beta=1, D_X=factory.D_X, train_size=100), slice(0, 50))
else:
    raise ValueError(f"Dataset {args.data} not found, available: toy, half_toy, linear")

bnn = factory.bnn()


if args.bald:
    fig, axs = plt.subplots(figsize=(18, 9), nrows=2, ncols=3)

    ax_prior, ax_hmc, ax_vi, ax_laplace, ax_swag, ax_balds = axs.ravel()
    for ax in axs.ravel()[1:-1]:
        ax.sharey(ax_prior)
    ax_balds.set_title("BALD scores")
    ax_balds.set_xlabel("x")
    ax_balds.set_ylabel("nats")

    def plot_experiment(experiment: Experiment, name, fig, ax):
        experiment.make_plots(fig=fig, ax=ax, plot_bald=False)
        ax.set_ylim(-6, 6)
        ax.set_title(name)
        bald_scores = experiment.compute_test_bald_scores()
        ax_balds.plot(experiment._data.test[0][:, 1], bald_scores, alpha=0.6, label=name)
else:
    fig = plt.figure(figsize=(18, 9))
    gs = gridspec.GridSpec(2, 6)

    axs = []

    for i in range(0, 5):
        if i < 3:
            if i == 0:
                ax = plt.subplot(gs[0, 2 * i:2 * i + 2])
            else:
                ax = plt.subplot(gs[0, 2 * i:2 * i + 2], sharey=axs[0])
                ax.sharey(axs[0])
                plt.setp(ax.get_yticklabels(), visible=False)
            ax.set_ylim(-6, 6)
        else:
            if i == 3:
                ax = plt.subplot(gs[1, 2 * i - 5:2 * i + 2 - 5])
            else:
                ax = plt.subplot(gs[1, 2 * i - 5:2 * i + 2 - 5])
                ax.sharey(axs[3])
                plt.setp(ax.get_yticklabels(), visible=False)
            ax.set_ylim(-6, 6)
        axs.append(ax)

    ax_prior, ax_hmc, ax_vi, ax_laplace, ax_swag = axs

    def plot_experiment(experiment: Experiment, name, fig, ax):
        t = experiment._data.test[0][:, 1]
        # Plot uncertainty quantiles
        experiment.make_plots(fig=fig, ax=ax)
        # Also plot function draws from posterior!
        for i in range(15):
            mu = experiment._predictions['Y_mean'][i, :, 0]
            sigma = experiment._predictions['Y_scale'][i, :, 0]
            ax.plot(t, mu, alpha=0.8)
            ax.fill_between(t, mu - sigma, mu + sigma, alpha=0.05)
        ax.set_title(name)
        ax.plot(data.train[0][:, 1], data.train[1][:, 0], "kx")


plot_prior_samples(bnn, data, fig=fig, ax=ax_prior, legend=True)
ax_prior.set_title("Prior")
if args.bald:
    prior_experiment = PriorExperiment(bnn, data)
    prior_experiment.make_predictions(random.PRNGKey(1))
    bald_scores = prior_experiment.compute_test_bald_scores()
    ax_balds.plot(data.test[0][:, 1], bald_scores, alpha=0.6, label="Prior")

delta = factory.map(bnn, data)
delta.train(random.PRNGKey(0))

deltas = [delta]
for i in range(1, factory.HMC_NUM_CHAINS):
    aux_delta = factory.map(bnn, data)
    aux_delta.train(random.PRNGKey(i))
    deltas.append(aux_delta)

hmc = factory.hmc(bnn, data, deltas)
hmc.train(random.PRNGKey(0))
del deltas
hmc.make_predictions(random.PRNGKey(1))

plot_experiment(hmc, "HMC", fig, ax_hmc)

vi = factory.mfvi(bnn, data)
bnn.BETA = factory.BETA
vi.train(random.PRNGKey(0))
bnn.BETA = 1.0
vi.make_predictions(random.PRNGKey(1))

plot_experiment(vi, "VI", fig, ax_vi)

laplace = factory.diag_laplace(bnn, data, delta)
laplace.train(random.PRNGKey(0))
laplace.make_predictions(random.PRNGKey(1))

plot_experiment(laplace, "Laplace", fig, ax_laplace)
ax_laplace.legend()
if not args.bald:
    # Add legend entry for mean+-std draws
    handles, labels = ax_laplace.get_legend_handles_labels()
    prior_legend = ax_prior.get_legend()
    ax_laplace.legend(handles + prior_legend._legend_elements_arg,
                      labels + [prior_legend.texts[0].get_text()],
                      handler_map=prior_legend.get_legend_handler_map())

swag = factory.swag(bnn, data, delta)
swag.train(random.PRNGKey(0))
swag.make_predictions(random.PRNGKey(1))

plot_experiment(swag, "SWAG", fig, ax_swag)

if args.bald:
    ax_balds.legend()
else:
    ax_prior.get_legend().remove()

fig.savefig(f"figs/complexity_comparison_{args.factory}_{args.data}_{args.D_X}{'_bald' if args.bald else ''}.pdf")
