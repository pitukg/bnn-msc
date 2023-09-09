import argparse
import pickle

import matplotlib.pyplot as plt
import numpyro
from jax import random
from numpyro.infer import log_likelihood

from experiments.src.data import *
from experiments.src.experiment import BasicHMCExperiment, AutoDiagonalLaplaceExperiment, Experiment
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
    prog='PosteriorLogliks',
    description='Cross-sectional logliks')
parser.add_argument('--factory', choices=['big', 'small'])
parser.add_argument('--numiter', type=int, default=10)
parser.add_argument('--x', type=float, default=0.0)
parser.add_argument('--algo', default='hmc')
parser.add_argument('--data', choices=['toy', 'half_toy', 'flat', 'linear', 'linear_flat', 'linear0pt5'])
parser.add_argument('--B', type=int, default=50_000)
# parser.add_argument('--D_X', type=int, default=2)  # Feature expansion dimension
args = parser.parse_args()

factory = getattr(experiments.src.factory, args.factory)

if args.data == "toy":
    data = ToyData1(feat_D_X=factory.D_X, train_size=100)
elif args.data == "half_toy":
    data = DataSlice(ToyData1(feat_D_X=factory.D_X, train_size=100), slice(0, 50))
elif args.data == "flat":
    data = DataSlice(ToyData1(feat_D_X=factory.D_X, gen_D_X=2, train_size=100), slice(0, 50))
elif args.data == "linear":
    data = DataSlice(LinearData(intercept=1, beta=1, D_X=factory.D_X, train_size=100), slice(0, 50))
elif args.data == "linear_flat":
    data = LinearData(intercept=0.0, beta=0.0, D_X=factory.D_X)
elif args.data == "linear0pt5":
    data = LinearData(intercept=0.5, beta=0.0, D_X=factory.D_X)
else:
    raise ValueError(f"Dataset {args.data} not found, available: toy, half_toy, flat, linear, linear_flat")

NUM_ITER = args.numiter
NUM_TEST_Ys = 100
X_CROSS_SECTION = args.x

Y_test = jnp.linspace(-6, 6, num=NUM_TEST_Ys)[:, jnp.newaxis]
data._X_test = jnp.hstack([jnp.full_like(Y_test, 0.), jnp.full_like(Y_test, X_CROSS_SECTION)])
data._Y_test = Y_test

bnn = factory.bnn()

total_num_samples = 0
concatenated_logliks = jnp.empty(shape=(0, NUM_TEST_Ys))

if args.algo == "hmc":
    hmc = factory.map_then_hmc(bnn, data)
    hmc._num_samples = 80

    for cnt in range(NUM_ITER):
        hmc.train(random.PRNGKey(0))
        hmc.make_predictions(random.PRNGKey(1))

        fig, ax = plt.subplots()
        hmc.make_plots(fig=fig, ax=ax)

        samples = hmc.get_posterior_samples()
        assert np.prod(jnp.shape(samples['w'])) > 0, "Experiment not trained"
        assert data.test[1] is not None, "Data does not contain test Y values"
        batch_ndims = 1 if not hmc._group_by_chain else 2
        logliks = log_likelihood(bnn, samples, batch_ndims=batch_ndims, X=data.test[0], Y=data.test[1])
        assert jnp.shape(logliks['Y'])[1] == jnp.shape(data.test[0])[0]
        num_samples = jnp.shape(logliks['Y'])[0]
        # logliks = logliks['Y'].sum(axis=1)

        total_num_samples += num_samples
        concatenated_logliks = jnp.append(concatenated_logliks, logliks['Y'], axis=0)

        # with open(f"preds{cnt}.pkl", "wb") as f:
        #     pickle.dump(hmc._predictions, f)
        hmc._samples["w"] = jnp.empty((0, bnn.get_weight_dim(),))
        hmc._predictions = None

else:
    Block = getattr(factory, args.algo)
    experiment = Block(bnn, data)
    experiment.train(random.PRNGKey(0))

    sample_rng_key = random.PRNGKey(9999)
    for _ in range(NUM_ITER*4):
        sample_rng_key, curr_rng_key = random.split(sample_rng_key)
        samples = experiment.get_posterior_samples(rng_key=curr_rng_key, num_samples=80)
        assert np.prod(jnp.shape(samples['w'])) > 0, "Experiment not trained"
        assert data.test[1] is not None, "Data does not contain test Y values"
        logliks = log_likelihood(bnn, samples, X=data.test[0], Y=data.test[1])
        assert jnp.shape(logliks['Y'])[1] == jnp.shape(data.test[0])[0]
        num_samples = jnp.shape(logliks['Y'])[0]

        total_num_samples += num_samples
        concatenated_logliks = jnp.append(concatenated_logliks, logliks['Y'], axis=0)


def get_logliks(ind_logliks):
    assert jnp.shape(ind_logliks) == (total_num_samples, NUM_TEST_Ys)
    return jax.scipy.special.logsumexp(ind_logliks, axis=0) - jnp.log(total_num_samples)


Y_logliks = get_logliks(concatenated_logliks)

B = args.B

ls = np.empty((B, NUM_TEST_Ys))
np.random.seed(0)
for b in range(B):
    ind_logliks = concatenated_logliks[np.random.choice(np.arange(total_num_samples), size=total_num_samples, replace=True)]
    ls[b, :] = get_logliks(ind_logliks)

ci = 2 * Y_logliks - np.quantile(ls, q=(0.9, 0.1), axis=0)

fig, ax = plt.subplots()
ax.plot(Y_test[..., 0], Y_logliks, color="darkblue", linewidth=1.5,
        label=f"Log-likelihood estimator\n({total_num_samples} samples)")
ax.fill_between(Y_test[..., 0], *ci, color="darkblue", alpha=0.5,
                label="90% bootstrap CI")
ax.legend()
ax.set_title(f"Posterior predictive at x={X_CROSS_SECTION:.1f}")
ax.set_xlabel("y")
ax.set_ylabel("log-likelihood")
fig.savefig(f"figs/posterior_logliks_{args.algo}_{args.data}_{args.factory}_2.pdf")
