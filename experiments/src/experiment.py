__all__ = [
    "Experiment",
    "PriorExperiment",
    "TrueBayesianLinearRegressionExperiment",
    "BasicHMCExperiment",
    "BasicSGLDExperiment",
    "AutoDeltaVIExperiment",
    "AutoMeanFieldNormalVIExperiment",
    "BasicMeanFieldGaussianVIExperiment",
    "InvGammaObservedMeanFieldGaussianVIExperiment",
    "BasicFullRankGaussianVIExperiment",
    "AutoFullRankLaplaceExperiment",
    "AutoFullRankLaplaceExperiment2",
    "AutoDiagonalLaplaceExperiment",
    "SWAGExperiment",
    "SequentialExperiment",
    "ExperimentWithLastBlockReplaced",
    "init_loc_fn",
    "plot_prior_samples",
]

import os
import pickle
import time
import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Type

import jax
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import numpyro
import numpyro.distributions as dist
import optax
import optax.second_order
import optax_swag
import tqdm
from jax import lax, vmap
from jax.scipy.special import logsumexp
from numpyro import handlers
from numpyro.distributions import constraints
from numpyro.infer import autoguide, log_likelihood, MCMC, NUTS, Predictive, SVI, TraceMeanField_ELBO, Trace_ELBO
from numpyro.infer.svi import SVIState
from numpyro.util import not_jax_tracer
from scipy import stats
from typing_extensions import Self

from .data import Data, DataSlice
from .model import BayesianNeuralNetwork, BayesianLinearRegression
from .sgld import SGLD


class Experiment(ABC):
    def __init__(self, bnn: BayesianNeuralNetwork, data: Data):
        self._bnn: BayesianNeuralNetwork = bnn
        self._data: Data = data
        # Initialise state
        self._predictions: Optional[dict] = None  # numpyro trace on data.test predictive
        # self._predictions: Optional[jnp.ndarray] = None  # of shape (num_samples, X_test.shape[0])

    @abstractmethod
    def train(self, rng_key_train: random.PRNGKey):
        pass

    @abstractmethod
    def make_predictions(self, rng_key_predict: random.PRNGKey):
        pass

    def make_plots(self, fig=None, ax=None, plot_bald=False, plot_samples=False, legend=False,
                   num_extend_samples=100, xlabel=True, ylabel=True, **kwargs) -> plt.Figure:
        assert self._predictions is not None
        X, Y = self._data.train
        X_test, _ = self._data.test
        # plotting
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        if self._bnn.OBS_MODEL != "classification":
            # compute mean prediction and confidence interval around median
            Y_mean_pred = self._predictions["Y_mean"]
            if self._bnn.OBS_MODEL == "const_prec":
                Y_scale = self._predictions["sigma_obs"]
                assert jnp.all(Y_scale[0] == Y_scale)
                Y_scale = Y_scale[0]
            elif self._bnn.OBS_MODEL == "inv_gamma":
                Y_scale = self._predictions["sigma_obs"][:, jnp.newaxis, jnp.newaxis]
            else:
                Y_scale = self._predictions["Y_scale"]
            z = np.random.randn(*(jnp.shape(Y_mean_pred)[:-1] + (num_extend_samples,)))
            extended_samples = Y_mean_pred + jnp.multiply(Y_scale, z)
            Y_pred = jnp.concatenate(jnp.swapaxes(extended_samples, axis1=1, axis2=2), axis=0)
            Y_mean_pred = Y_mean_pred[..., 0]
            mean_means = jnp.mean(Y_mean_pred, axis=0)
            mean_percentiles = np.percentile(Y_mean_pred, [5.0, 95.0], axis=0)
            Y_percentiles = np.percentile(Y_pred, [5.0, 95.0], axis=0)
            # Plot uncertainty regions
            ax.fill_between(X_test[:, 1], *mean_percentiles, color="#ffae22", alpha=0.5, label="90% HPDI on mean")
            ax.fill_between(X_test[:, 1], Y_percentiles[0], mean_percentiles[0], color="#2273ff", alpha=0.5, label="90% HPDI on prediction")
            ax.fill_between(X_test[:, 1], mean_percentiles[1], Y_percentiles[1], color="#2273ff", alpha=0.5)
            if plot_samples:
                # Plot function draws from Y_mean
                label_flag = True
                for i in range(-30, 0, 2):  # Plot from last => mixes in case of HMC
                    label = "Posterior function draws" if label_flag else None
                    label_flag = False
                    ax.plot(X_test[:, 1], Y_mean_pred[i], color="black", linewidth=0.5, alpha=0.8, label=label)
            # plot mean
            ax.plot(X_test[:, 1], mean_means, color="darkblue", linewidth=1.5, label="Posterior mean")
            # plot training data
            ax.plot(X[:, 1], Y[:, 0], "kx")
            if plot_bald:
                raise NotImplementedError()  # Shouldn't mix axis units
                bald_scores = self.compute_test_bald_scores()
                ax.plot(X_test[:, 1], bald_scores, color="black", alpha=0.6, label="BALD score")
            if xlabel:
                ax.set_xlabel("x")
            if ylabel:
                ax.set_ylabel("y")
            if legend:
                ax.legend()
        else:
            percentiles90 = jnp.percentile(self._predictions['Y_p'][:, :, 0, 1], q=jnp.array([5.0, 95.0]), axis=0)
            percentiles50 = jnp.percentile(self._predictions['Y_p'][:, :, 0, 1], q=jnp.array([25.0, 75.0]), axis=0)
            ax.plot(X_test[:, 1], self._predictions['Y_p'][:, :, 0, 1].mean(axis=0))
            ax.fill_between(X_test[:, 1], *percentiles90, color='orange', alpha=0.3)
            ax.fill_between(X_test[:, 1], *percentiles50, color='orange', alpha=0.3)
            ax.plot(X_test[:, 1], self._data.true_predictive(X_test[:, 1]).mean, linestyle=':', color="black")
            ax.xaxis.set_minor_locator(matplotlib.ticker.FixedLocator(X[:, 1]))
        return fig

    def run(self, rng_key: random.PRNGKey):
        rng_key_train, rng_key_predict = random.split(rng_key)
        self.train(rng_key_train)
        self.make_predictions(rng_key_predict)
        fig = self.make_plots()
        return fig

    @abstractmethod
    def get_posterior_samples(self, **kwargs) -> dict:
        # Should be keyed by w and have dimensions (num_samples, p)
        raise NotImplementedError()

    def test_loglik(self) -> float:
        """ Calculate loglikelihood of test observations, based on the trained experiment's posterior.
            This is a common metric for assessing generalisation and comparing performance of Bayesian models.
        :return: posterior log-likelihood of observations in test
        """
        samples = self.get_posterior_samples()
        assert np.prod(jnp.shape(samples['w'])) > 0, "Experiment not trained"
        assert self._data.test[1] is not None, "Data does not contain test Y values"
        logliks = log_likelihood(self._bnn, samples, X=self._data.test[0], Y=self._data.test[1])
        assert jnp.shape(logliks['Y'])[1] == jnp.shape(self._data.test[0])[0]
        num_samples = jnp.shape(logliks['Y'])[0]
        logliks = logliks['Y'].sum(axis=1)
        return logsumexp(logliks) - jnp.log(num_samples)

    def compute_test_bald_scores(self, rng_key=random.PRNGKey(0)):
        assert self._predictions is not None, "Experiment not trained"
        if self._bnn.OBS_MODEL != "classification":
            # H[ y | x_test, D ]
            Y_pred = self._predictions["Y"][..., 0]
            posterior_Y_entropy = stats.differential_entropy(Y_pred, axis=0)
            # E_{w ~ p(.|D)} H[ y | x_test, w ]
            samples = self.get_posterior_samples()
            num_samples = jnp.shape(samples["w"])[0]

            def gaussian_entropy(locs, scales):
                return jnp.log(scales) + 0.5 * (jnp.log(2. * jnp.pi) + 1.)

            def cond_entropy(sample, rng_key):
                conditioned_model = handlers.condition(handlers.seed(self._bnn, rng_key), sample)
                trace = handlers.trace(conditioned_model).get_trace(X=self._data.test[0], Y=None)
                conditional_distribution = trace["Y"]["fn"].base_dist
                return gaussian_entropy(conditional_distribution.loc, conditional_distribution.scale)

            Y_cond_entropies = vmap(cond_entropy)(samples, random.split(rng_key, num_samples))[..., 0]
            return posterior_Y_entropy - jnp.mean(Y_cond_entropies, axis=0)

        else:
            raise NotImplementedError()

    def __getstate__(self) -> dict:
        return {
            k: v
            for k, v in self.__dict__.items()
            if k not in ("_bnn", "_data", "_predictions", "_mcmc", "_svi", "_saved_svi_state", "_guide", "_lr_schedule")
        }

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._bnn = None
        self._data = None
        self._predictions = None

    def to_pickle(self, filename: str):
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def from_pickle(bnn: BayesianNeuralNetwork, data: Data, filename: str):
        with open(filename, "rb") as f:
            experiment = pickle.load(f)
        experiment._bnn = bnn
        experiment._data = data
        return experiment


class SequentialExperimentBlock(Experiment):
    @property
    @abstractmethod
    def posterior(self) -> tuple[dist.Distribution, Optional[dist.Distribution]]:
        """ Returns distribution on w and prec_obs """
        raise NotImplementedError()

    def get_posterior_samples(self, rng_key=random.PRNGKey(0), num_samples=400, **kwargs) -> dict:
        w_posterior = self.posterior[0]
        samples = w_posterior.sample(rng_key, (num_samples,))
        assert jnp.shape(samples) == (num_samples, self._bnn.get_weight_dim())
        return {'w': samples}


class PriorExperiment(Experiment):
    def train(self, rng_key_train: random.PRNGKey):
        pass

    def make_predictions(self, rng_key_predict: random.PRNGKey):
        predictive = Predictive(self._bnn, num_samples=self.num_samples)
        self._predictions = predictive(rng_key_predict, X=self._data.test[0], Y=None)

    def get_posterior_samples(self, **kwargs) -> dict:
        return {'w': self._predictions['w']}

    def __init__(self, bnn: BayesianNeuralNetwork, data: Data, num_samples: int = 400):
        super().__init__(bnn, data)
        self.num_samples = num_samples


class TrueBayesianLinearRegressionExperiment(SequentialExperimentBlock):
    def __init__(self, bnn: BayesianLinearRegression, data: Data, num_samples: int = 400):
        self._bnn: BayesianLinearRegression = bnn
        self._data: Data = data
        self.num_samples = num_samples
        # Initialise state
        self._predictions: Optional[dict] = None  # numpyro trace on data.test predictive
        # self._predictions: Optional[jnp.ndarray] = None  # of shape (num_samples, X_test.shape[0])
        self._bnn_post: Optional[BayesianLinearRegression] = None

    def train(self, rng_key_train: random.PRNGKey):
        X, Y = self._data.train
        mu, V = self._bnn._prior_w.mean, self._bnn._prior_w.covariance_matrix
        prec_obs = self._bnn._prior_prec_obs.mean
        V_inv = jnp.linalg.inv(V)
        V_post_inv = V_inv + prec_obs * X.T @ X
        V_post = jnp.linalg.inv(V_post_inv)
        mu_post = V_post @ (V_inv @ mu + prec_obs * X.T @ Y[:, 0])
        self._bnn_post = self._bnn.with_prior(dist.MultivariateNormal(mu_post, V_post), self._bnn._prior_prec_obs)

    def make_predictions(self, rng_key_predict: random.PRNGKey):
        predictive = Predictive(self._bnn_post, num_samples=self.num_samples)
        self._predictions = predictive(rng_key_predict, X=self._data.test[0], Y=None)

    def get_posterior_samples(self, **kwargs) -> dict:
        return {'w': self._predictions['w']}

    @property
    def posterior(self) -> tuple[dist.Distribution, Optional[dist.Distribution]]:
        assert self._bnn_post is not None
        # w_posterior = dist.Normal(loc=self._bnn_post._prior_w.mean, scale=self._bnn_post._prior_w.variance).to_event(1)
        w_posterior = self._bnn_post._prior_w
        # Note for further VI it is a problem that support(prec_obs) is a single point,
        # therefore we mask this distribution so KL computation is ignored, and make sure to
        # initialise the delta guide to this point!
        prec_obs_posterior = self._bnn_post._prior_prec_obs
        # prec_obs_posterior = dist.Delta(self._params["prec_obs_loc"]).mask(False) \
        #     if "prec_obs_loc" in self._params.keys() else None
        return w_posterior, prec_obs_posterior


class BasicHMCExperiment(Experiment):
    def __init__(self, bnn: BayesianNeuralNetwork, data: Data, init_params: Optional[dict] = None,
                 num_samples: int = 2_000, num_warmup: int = 1_000, thinning=1,
                 num_chains: int = 1, group_by_chain: bool = False):
        super().__init__(bnn, data)
        self._init_params = init_params
        self._num_samples = num_samples
        self._num_warmup = num_warmup
        self._thinning = thinning
        self._num_chains = num_chains
        self._group_by_chain = group_by_chain
        # Initialise state
        self._mcmc = None
        samples_init_shape = (0, self._bnn.get_weight_dim(),) if not self._group_by_chain else \
            (num_chains, 0, self._bnn.get_weight_dim(),)
        self._samples: dict = dict(w=jnp.empty(samples_init_shape))

    def _get_kernel(self):
        return NUTS(self._bnn)

    def train(self, rng_key_train: random.PRNGKey, progress_bar: bool = True):
        start = time.time()
        X, Y = self._data.train
        if self._mcmc is None:
            kernel = self._get_kernel()
            self._mcmc = MCMC(
                kernel,
                num_warmup=self._num_warmup,
                num_samples=self._num_samples,
                thinning=self._thinning,
                num_chains=self._num_chains,
                chain_method="vectorized",
                progress_bar=False if not progress_bar or "NUMPYRO_SPHINXBUILD" in os.environ else True,
            )
        else:
            self._mcmc.progress_bar = progress_bar
            if self._mcmc.post_warmup_state is not None:
                rng_key_train = self._mcmc.post_warmup_state.rng_key
        self._mcmc.run(rng_key_train, init_params=self._init_params, X=X, Y=Y)
        if progress_bar:
            # mcmc.print_summary()
            print("\nMCMC elapsed time:", time.time() - start)
        self._samples['w'] = jnp.concatenate(
            (self._samples['w'], self._mcmc.get_samples(group_by_chain=self._group_by_chain)['w']),
            axis=0 if not self._group_by_chain else 1
        )
        self._mcmc.post_warmup_state = self._mcmc.last_state

    def make_predictions(self, rng_key_predict: random.PRNGKey):
        assert self._samples is not None
        X_test, _ = self._data.test
        if not self._group_by_chain:
            self._predictions = Predictive(
                self._bnn, self._samples, return_sites=['Y_mean', 'Y_scale', 'Y_p', 'Y', 'sigma_obs'])(
                rng_key_predict, X=X_test, Y=None)  # ['Y'][..., 0]
        else:
            def pred(rng_key, samples):
                return Predictive(self._bnn, samples)(rng_key, X=X_test, Y=None)

            self._predictions = vmap(pred)(random.split(rng_key_predict, self._num_chains), self._samples)

    def test_loglik(self) -> float:
        assert np.prod(jnp.shape(self._samples['w'])) > 0, "HMC not trained"
        assert self._data.test[1] is not None, "Data does not contain test Y values"
        batch_ndims = 1 if not self._group_by_chain else 2
        logliks = log_likelihood(self._bnn, self._samples, batch_ndims=batch_ndims,
                                 X=self._data.test[0], Y=self._data.test[1])
        assert jnp.shape(logliks['Y'])[1] == jnp.shape(self._data.test[0])[0]
        num_samples = jnp.shape(logliks['Y'])[0]
        logliks = logliks['Y'].sum(axis=1)
        return logsumexp(logliks) - jnp.log(num_samples)

    def get_posterior_samples(self) -> dict:
        if self._group_by_chain:
            raise NotImplementedError()
        return self._samples


class BasicSGLDExperiment(BasicHMCExperiment):
    def __init__(self, bnn: BayesianNeuralNetwork, data: Data, init_params: Optional[dict] = None,
                 init_step_size = 0.1, final_step_size = 0.0001, num_samples: int = 10_000,
                 num_warmup: int = 5_000, thinning: int = 5,
                 num_chains: int = 1, group_by_chain: bool = False):
        super().__init__(bnn, data, init_params, num_samples, num_warmup, thinning, num_chains, group_by_chain)
        self._init_step_size = init_step_size
        self._final_step_size = final_step_size

    def _constant_lr_schedule_with_cosine_burnin(self, step):
        """
        Cosine LR schedule with burn-in for SG-MCMC.
        Based on [bnn_hmc](https://github.com/google-research/google-research/tree/master/bnn_hmc).
        """
        t = jnp.minimum(step / self._num_warmup, 1.)
        coef = (1 + jnp.cos(t * np.pi)) * 0.5
        return coef * self._init_step_size + (1 - coef) * self._final_step_size

    def _get_kernel(self):
        return SGLD(
            self._bnn,
            init_strategy=numpyro.infer.init_to_sample,
            step_size_fn=lambda _: self._constant_lr_schedule_with_cosine_burnin,
        )


class EvalLoss:
    def __init__(self, num_particles=1):
        self.num_particles = num_particles

    def loss(self, rng_key, param_map, model, guide, *args, **kwargs):
        def single_particle_elbo(rng_key):
            params = param_map.copy()
            model_seed, guide_seed = random.split(rng_key)
            seeded_model = handlers.seed(model, model_seed)
            seeded_guide = handlers.seed(guide, guide_seed)
            subs_guide = handlers.substitute(seeded_guide, data=param_map)
            guide_trace = handlers.trace(subs_guide).get_trace(*args, **kwargs)
            subs_model = handlers.substitute(handlers.replay(seeded_model, guide_trace), data=params)
            model_trace = handlers.trace(subs_model).get_trace(*args, **kwargs)
            # check_model_guide_match(model_trace, guide_trace)
            # _validate_model(model_trace, plate_warning="loose")
            # _check_mean_field_requirement(model_trace, guide_trace)

            elbo_lik = 0
            elbo_kl = 0
            for name, model_site in model_trace.items():
                if model_site["type"] == "sample":
                    if model_site["is_observed"]:
                        elbo_lik = elbo_lik + numpyro.infer.elbo._get_log_prob_sum(model_site)
                    else:
                        guide_site = guide_trace[name]
                        try:
                            kl_qp = dist.kl.kl_divergence(guide_site["fn"], model_site["fn"])
                            if guide_site["scale"] is not None:
                                kl_qp = kl_qp * guide_site["scale"]
                            elbo_kl = elbo_kl + jnp.sum(kl_qp)
                            # elbo_lik = elbo_lik - jnp.sum(kl_qp)
                        except NotImplementedError:
                            # raise NotImplementedError()
                            elbo_kl = (
                                    elbo_kl
                                    + numpyro.infer.elbo._get_log_prob_sum(guide_site)
                                    - numpyro.infer.elbo._get_log_prob_sum(model_site)
                            )

            # handle auxiliary sites in the guide
            for name, site in guide_trace.items():
                if site["type"] == "sample" and name not in model_trace:
                    assert site["infer"].get("is_auxiliary") or site["is_observed"]
                    elbo_lik = elbo_lik - numpyro.infer.elbo._get_log_prob_sum(site)

            return elbo_lik, elbo_kl

        if self.num_particles == 1:
            elbo_lik, elbo_kl = single_particle_elbo(rng_key)
            return {"elbo_lik": elbo_lik, "elbo_kl": elbo_kl, "loss": -elbo_lik + elbo_kl}
        else:
            rng_keys = random.split(rng_key, self.num_particles)
            elbo_liks, elbo_kls = vmap(single_particle_elbo)(rng_keys)
            assert jnp.all(elbo_kls == elbo_kls[0])
            return {
                "elbo_lik": jnp.mean(elbo_liks),
                "elbo_kl": elbo_kls[0],
                "loss": -jnp.mean(elbo_liks) + elbo_kls[0]
            }


class BasicVIExperiment(SequentialExperimentBlock):
    def __init__(self, bnn: BayesianNeuralNetwork, data: Data, num_samples: int = 2_000,
                 max_iter: int = 150_000, lr_schedule: optax.Schedule = optax.constant_schedule(-0.001),
                 num_particles: int = 16, num_eval_particles: int = 128):
        super().__init__(bnn, data)
        self._num_samples = num_samples
        self._max_iter = max_iter
        self._lr_schedule = lr_schedule  # Note should be negative!
        self._num_particles = num_particles
        self._num_eval_particles = num_eval_particles
        # Initialise state
        self._svi: Optional[SVI] = None
        self._guide: Optional[Callable] = None
        self._saved_svi_state: Optional[SVIState] = None
        self._losses: jnp.array = jnp.array([])
        self._eval_losses: jnp.array = jnp.array([]).reshape((0, 3))
        self._params: Optional[dict] = None

    def train(self, rng_key_train: random.PRNGKey, num_iter: Optional[int] = None):
        if num_iter is None:
            num_iter = self._max_iter

        start = time.time()
        X, Y = self._data.train

        if self._svi is None:
            self._guide = self._get_guide()
            # Custom optimizer to prevent effect of exploding gradients (by tail ELBO estimates)
            clipped_adam = optax.chain(optax.clip_by_global_norm(10.0),
                                       optax.scale_by_adam(),
                                       optax.scale_by_schedule(self._lr_schedule))
            optimizer = clipped_adam  # Default taken from ashleve/lightning-hydra-template
            train_loss = TraceMeanField_ELBO(num_particles=self._num_particles)
            self._svi = SVI(self._bnn, self._guide, optimizer, train_loss)
        eval_loss = EvalLoss(num_particles=self._num_eval_particles)
        rng_key_train, rng_key_eval, rng_key_init_loss = random.split(rng_key_train, 3)

        if self._saved_svi_state is None:
            # If restoring from pickle _params will be set, so we initialize from there
            # otherwise _params will be None and this will have no effect
            self._saved_svi_state = self._svi.init(rng_key_train, init_params=self._params, X=X, Y=Y)

        def body_fn(svi_state, _):
            svi_state, loss = self._svi.stable_update(svi_state, X=X, Y=Y)
            return svi_state, loss

        init_eval_loss = eval_loss.loss(
            rng_key_init_loss, self._svi.get_params(self._saved_svi_state), self._bnn, self._guide, X=X, Y=Y)
        traced = not not_jax_tracer(init_eval_loss["loss"])
        if not traced:
            print("Initial eval loss: {:.4f} (lik: {:.4f}, kl: {:.4f})".format(
                init_eval_loss["loss"], init_eval_loss["elbo_lik"], init_eval_loss["elbo_kl"]))

        batch = max(num_iter // 50, 1)
        with tqdm.trange(1, num_iter // batch + 1, disable=traced) as t:
            for i in t:
                self._saved_svi_state, batch_losses = lax.scan(body_fn, self._saved_svi_state, None, length=batch)
                self._losses = jnp.concatenate((self._losses, batch_losses))
                valid_losses = [x for x in batch_losses if traced or x == x]
                num_valid = len(valid_losses)
                if num_valid == 0:
                    avg_loss = float("nan")
                else:
                    avg_loss = sum(valid_losses) / num_valid
                # Compute full loss
                rng_key_eval, rng_key_eval_curr = random.split(rng_key_eval)
                eval_loss_res = eval_loss.loss(rng_key_eval_curr, self._svi.get_params(self._saved_svi_state),
                                               self._bnn, self._guide, X=X, Y=Y)
                self._eval_losses = jnp.append(
                    self._eval_losses,
                    jnp.array([[eval_loss_res["loss"], eval_loss_res["elbo_lik"], eval_loss_res["elbo_kl"]]]),
                    axis=0
                )
                if not traced:
                    t.set_postfix_str(
                        "init loss: {:.4f}, avg. train loss / eval. loss [{}-{}]: {:.4f} / {:.4f}".format(
                            self._losses[0], (i - 1) * batch, i * batch, avg_loss, eval_loss_res["loss"]
                        ),
                        refresh=False,
                    )
        self._params = self._svi.get_params(self._saved_svi_state)
        if not traced:
            print("\nSVI elapsed time:", time.time() - start)

    def make_predictions(self, rng_key_predict: random.PRNGKey):
        assert self._params is not None
        X_test, _ = self._data.test
        predictive = Predictive(model=self._bnn, guide=self._guide,
                                params=self._params, num_samples=self._num_samples)
        self._predictions = predictive(rng_key_predict, X=X_test, Y=None)  # ['Y'][..., 0]

    def show_convergence_plot(self, fig=None, ax=None) -> plt.Figure:
        if fig is None or ax is None:
            fig, ax = plt.subplots()
        ax.plot(self._eval_losses[:, 0], label="loss")
        ax.plot(self._eval_losses[:, 1], label="lik")
        ax.plot(-self._eval_losses[:, 2], label="-kl")
        return fig

    @property
    @abstractmethod
    def posterior(self) -> tuple[dist.Distribution, Optional[dist.Distribution]]:
        """ :returns distribution of w and (prec_obs if in the model)
            Note if prec_obs has a Delta distribution, it should be marked as masked so that
            hack with keeping it constant under another Delta approximation doesn't blow up loss
        """
        raise NotImplementedError()

    @abstractmethod
    def _get_guide(self) -> Callable[[jnp.ndarray, Optional[jnp.ndarray]], Any]:
        # This needs to enforce that if self._bnn's prior on prec_obs is masked then
        # in the guide, prec_obs is treated as a constant and not as a numpy.param + Delta
        # so that gradients exist and loss is not inf
        raise NotImplementedError()

    def get_posterior_samples(self, rng_key=random.PRNGKey(0), **kwargs) -> dict:
        return super().get_posterior_samples(rng_key, self._num_samples)

    @staticmethod
    def from_pickle(bnn: BayesianNeuralNetwork, data: Data, filename: str):
        experiment = Experiment.from_pickle(bnn, data, filename)
        experiment._svi = None
        experiment._saved_svi_state = None
        experiment._guide = None
        experiment._lr_schedule = optax.constant_schedule(-0.005)  # Can't be pickled
        # Make sure state is initialised
        experiment.train(random.PRNGKey(0), num_iter=0)
        return experiment


class BasicMeanFieldGaussianVIExperiment(BasicVIExperiment):
    def __init__(self, bnn: BayesianNeuralNetwork, data: Data, num_samples: int = 2_000,
                 max_iter: int = 150_000, lr_schedule: optax.Schedule = optax.constant_schedule(-0.001),
                 num_particles: int = 16, num_eval_particles: int = 128, init_scale: float = 1e-5):
        super(BasicMeanFieldGaussianVIExperiment, self).__init__(bnn,data, num_samples, max_iter, lr_schedule, num_particles, num_eval_particles)
        self.init_scale = init_scale

    def _get_guide(self) -> Callable[[jnp.ndarray, Optional[jnp.ndarray]], Any]:
        bnn_weight_dim = self._bnn.get_weight_dim()
        bnn_weight_prior = self._bnn.prior[0]

        def guide(X, Y=None):
            w_loc = numpyro.param("w_loc", lambda rng_key: bnn_weight_prior.sample(rng_key))
            w_scale = numpyro.param("w_scale", jnp.full((bnn_weight_dim,), self.init_scale),
                                    constraint=constraints.softplus_positive)
            with handlers.scale(scale=self._bnn.BETA):
                numpyro.sample("w", dist.Normal(w_loc, w_scale).to_event(1))
            _, prec_obs_prior = self._bnn.prior
            if prec_obs_prior is not None:
                # See comment above for initialising prec_obs to its point mass as it is masked!
                # Taking the prior mean returns the delta mass location in the Delta case
                prec_obs_loc = numpyro.param("prec_obs_loc", prec_obs_prior.mean, constraint=constraints.positive)
                prec_obs_dist = dist.Delta(prec_obs_loc)
                if isinstance(prec_obs_prior, dist.MaskedDistribution):
                    # Treat prec_obs as constant here, decouple from parameter completely,
                    # otherwise it would give MAP on uniform improper prior
                    del prec_obs_dist  # Lose dependence on "prec_obs_loc" numpyro.param
                    prec_obs_dist = dist.Delta(prec_obs_prior.mean)
                numpyro.sample("prec_obs", prec_obs_dist)

        return guide

    @property
    def posterior(self) -> tuple[dist.Distribution, Optional[dist.Distribution]]:
        assert self._params is not None
        w_posterior = dist.Normal(loc=self._params["w_loc"], scale=self._params["w_scale"]).to_event(1)
        # Note for further VI it is a problem that support(prec_obs) is a single point,
        # therefore we mask this distribution so KL computation is ignored, and make sure to
        # initialise the delta guide to this point!
        prec_obs_posterior = dist.Delta(self._params["prec_obs_loc"]).mask(False) \
            if "prec_obs_loc" in self._params.keys() else None
        return w_posterior, prec_obs_posterior


class InvGammaObservedMeanFieldGaussianVIExperiment(BasicVIExperiment):
    def _get_guide(self) -> Callable[[jnp.ndarray, Optional[jnp.ndarray]], Any]:
        bnn_weight_dim = self._bnn.get_weight_dim()
        bnn_weight_prior = self._bnn.prior[0]

        def guide(X, Y=None):
            w_loc = numpyro.param("w_loc", lambda rng_key: bnn_weight_prior.sample(rng_key))
            w_scale = numpyro.param("w_scale", jnp.full((bnn_weight_dim,), 1e-5),
                                    constraint=constraints.softplus_positive)
            with handlers.scale(scale=self._bnn.BETA):
                numpyro.sample("w", dist.Normal(w_loc, w_scale).to_event(1))
            _, prec_obs_prior = self._bnn.prior
            assert isinstance(prec_obs_prior, dist.Gamma), "Expecting Inverse Gaussian observation variance"
            # Model posterior observation variance as an independent inverse Gamma again (initialized to prior)
            prec_obs_concentration = numpyro.param("prec_obs_concentration", prec_obs_prior.concentration,
                                                   constraint=constraints.positive)
            prec_obs_rate = numpyro.param("prec_obs_rate", prec_obs_prior.rate,
                                          constraint=constraints.positive)
            prec_obs_dist = dist.Gamma(prec_obs_concentration, prec_obs_rate)
            numpyro.sample("prec_obs", prec_obs_dist)

        return guide

    @property
    def posterior(self) -> tuple[dist.Distribution, Optional[dist.Distribution]]:
        assert self._params is not None
        w_posterior = dist.Normal(loc=self._params["w_loc"], scale=self._params["w_scale"]).to_event(1)
        prec_obs_posterior = dist.Gamma(concentration=self._params["prec_obs_concentration"],
                                        rate=self._params["prec_obs_rate"])
        return w_posterior, prec_obs_posterior


class AutoMeanFieldNormalVIExperiment(BasicVIExperiment):
    def _get_guide(self) -> Callable:
        return autoguide.AutoNormal(self._bnn, init_loc_fn=numpyro.infer.init_to_sample, init_scale=1e-5)

    @property
    def posterior(self) -> tuple[dist.Distribution, dist.Distribution]:
        raise NotImplementedError()


class AutoDeltaVIExperiment(BasicVIExperiment):
    def __init__(self, bnn: BayesianNeuralNetwork, data: Data, max_iter: int = 150_000,
                 lr_schedule: optax.Schedule = optax.constant_schedule(-0.001), num_samples: int = 200):
        # If KL computed analytically, things are deterministic -> single sample enough for loss
        # Hack for SWAG posterior:
        num_particles = 1
        if isinstance(bnn.prior[0], dist.LowRankMultivariateNormal):
            num_particles = 256
        super().__init__(bnn, data, num_samples=num_samples, max_iter=max_iter, lr_schedule=lr_schedule,
                         num_particles=num_particles, num_eval_particles=num_particles)

    def _get_guide(self) -> Callable:
        bnn_weight_dim = self._bnn.get_weight_dim()
        bnn_weight_prior = self._bnn.prior[0]
        _, prec_obs_prior = self._bnn.prior

        def guide(X, Y=None):
            w_loc = numpyro.param("w_loc", lambda rng_key: bnn_weight_prior.sample(rng_key) * 1.4)
            with handlers.scale(scale=self._bnn.BETA):
                numpyro.sample("w", dist.Delta(w_loc).to_event(1))
            if prec_obs_prior is not None:
                # See comment above for initialising prec_obs to its point mass as it is masked!
                # Taking the prior mean returns the delta mass location in the Delta case
                prec_obs_loc = numpyro.param("prec_obs_loc", prec_obs_prior.mean, constraint=constraints.positive)
                prec_obs_dist = dist.Delta(prec_obs_loc)
                if isinstance(prec_obs_prior, dist.MaskedDistribution):
                    # Treat prec_obs as constant here, decouple from parameter completely,
                    # otherwise it would give MAP on uniform improper prior
                    del prec_obs_dist  # Lose dependence on "prec_obs_loc" numpyro.param
                    prec_obs_dist = dist.Delta(prec_obs_prior.mean)
                numpyro.sample("prec_obs", prec_obs_dist)

        return guide
        # return autoguide.AutoDelta(self._bnn, init_loc_fn=init_loc_fn)

    @property
    def posterior(self) -> tuple[dist.Distribution, Optional[dist.Distribution]]:
        return dist.Delta(self._params['w_loc']), None


class BasicFullRankGaussianVIExperiment(BasicVIExperiment):
    def _get_guide(self) -> Callable[[jnp.ndarray, Optional[jnp.ndarray]], Any]:
        bnn_weight_dim = self._bnn.get_weight_dim()
        bnn_weight_prior = self._bnn.prior[0]

        def guide(X, Y=None):
            w_loc = numpyro.param("w_loc", lambda rng_key: bnn_weight_prior.sample(rng_key))
            w_cov = numpyro.param("w_cov", 0.1 * jnp.eye(bnn_weight_dim), constraint=constraints.positive_definite)
            with handlers.scale(scale=self._bnn.BETA):
                numpyro.sample("w", dist.MultivariateNormal(w_loc, w_cov))
            _, prec_obs_prior = self._bnn.prior
            if prec_obs_prior is not None:
                # See comment above for initialising prec_obs to its point mass as it is masked!
                # Taking the prior mean returns the delta mass location in the Delta case
                prec_obs_loc = numpyro.param("prec_obs_loc", prec_obs_prior.mean, constraint=constraints.positive)
                prec_obs_dist = dist.Delta(prec_obs_loc)
                if isinstance(prec_obs_prior, dist.MaskedDistribution):
                    # Treat prec_obs as constant here, decouple from parameter completely,
                    # otherwise it would give MAP on uniform improper prior
                    del prec_obs_dist  # Lose dependence on "prec_obs_loc" numpyro.param
                    prec_obs_dist = dist.Delta(prec_obs_prior.mean)
                numpyro.sample("prec_obs", prec_obs_dist)

        return guide

    @property
    def posterior(self) -> tuple[dist.Distribution, Optional[dist.Distribution]]:
        assert self._params is not None
        w_posterior = dist.MultivariateNormal(loc=self._params["w_loc"],
                                              covariance_matrix=self._params["w_cov"])
        # Note for further VI it is a problem that support(prec_obs) is a single point,
        # therefore we mask this distribution so KL computation is ignored, and make sure to
        # initialise the delta guide to this point!
        prec_obs_posterior = dist.Delta(self._params["prec_obs_loc"]).mask(False) \
            if "prec_obs_loc" in self._params.keys() else None
        return w_posterior, prec_obs_posterior


class AutoFullRankLaplaceExperiment(BasicVIExperiment):
    def __init__(self, bnn: BayesianNeuralNetwork, data: Data, shrink: float = 25.0, num_samples: int = 2_000,
                 max_iter: int = 150_000, lr_schedule: optax.Schedule = optax.constant_schedule(-0.001),
                 num_particles: int = 16, num_eval_particles: int = 128):
        super().__init__(bnn, data, num_samples, max_iter, lr_schedule, num_particles, num_eval_particles)
        self._shrink = shrink

    def _get_guide(self) -> Callable:
        self._guide = autoguide.AutoLaplaceApproximation(
            self._bnn, init_loc_fn=init_loc_fn,
            hessian_fn=lambda f, x: (jax.hessian(f)(x) + jnp.eye(x.shape[-1]) * self._shrink) * self._data.train[0].shape[0]
        )
        return self._guide

    @property
    def posterior(self) -> tuple[dist.Distribution, Optional[dist.Distribution]]:
        return self._guide.get_posterior(self._params), None

    def make_predictions(self, rng_key_predict: random.PRNGKey):
        assert self._params is not None and self._guide is not None
        X_test, _ = self._data.test
        posterior = self._guide.get_posterior(self._params)
        samples = posterior.sample(rng_key_predict, sample_shape=(self._num_samples,))
        predictive = Predictive(model=self._bnn, posterior_samples={'w': samples[:, :-1]})
        self._predictions = predictive(rng_key_predict, X=X_test, Y=None)  # ['Y'][..., 0]


class AutoDiagonalLaplaceApproximation(autoguide.AutoLaplaceApproximation):
    def __init__(
            self,
            model,
            *,
            prefix="auto",
            init_loc_fn=autoguide.init_to_uniform,
            create_plates=None,
            hessian_diag_fn=None,
    ):
        super().__init__(
            model, prefix=prefix, init_loc_fn=init_loc_fn, create_plates=create_plates
        )
        self._hessian_diag_fn = (
            hessian_diag_fn if hessian_diag_fn is not None else (
                lambda f, x: optax.hessian_diag(lambda params, _, __: f(params), x, None, None))
        )

    def get_transform(self, params):
        def loss_fn(z):
            params1 = params.copy()
            params1["{}_loc".format(self.prefix)] = z
            return self._loss_fn(params1)

        loc = params["{}_loc".format(self.prefix)]
        precision = self._hessian_diag_fn(loss_fn, loc)
        scale = 1. / jnp.sqrt(precision)
        if numpyro.util.not_jax_tracer(scale):
            if np.any(np.isnan(scale)):
                warnings.warn(
                    "Hessian of log posterior at the MAP point is singular. Posterior"
                    " samples from AutoLaplaceApproxmiation will be constant (equal to"
                    " the MAP point). Please consider using an AutoNormal guide.",
                    stacklevel=numpyro.util.find_stack_level(),
                )
        scale = jnp.where(jnp.isnan(scale), 0.0, scale)
        return dist.transforms.AffineTransform(loc, scale)

    def get_posterior(self, params):
        """
        Returns an isotropic Normal posterior distribution.
        """
        transform = self.get_transform(params)
        return dist.Normal(transform.loc, scale=transform.scale).to_event(1)


init_loc_fn = numpyro.infer.init_to_sample


class AutoDiagonalLaplaceExperiment(SequentialExperimentBlock):
    def __init__(self, bnn: BayesianNeuralNetwork, data: Data, trained_map_experiment: AutoDeltaVIExperiment,
                 shrink: float = 25.0, num_samples: int = 400):
        super().__init__(bnn, data)
        self._map_experiment = trained_map_experiment
        self._shrink = shrink
        self._num_samples = num_samples
        self._posterior = None

    def train(self, rng_key_train: random.PRNGKey):
        loss = TraceMeanField_ELBO(num_particles=1)
        rng_key = random.PRNGKey(0)

        def negative_loglik(params, inputs, targets):
            return loss.loss(rng_key, params, self._bnn, self._map_experiment._guide, X=inputs, Y=targets)

        X, Y = self._data.train
        # precision = optax.second_order.fisher_diag(negative_loglik, self._map_experiment._params, X, Y)
        # precision = precision[:-1]
        precision = jax.hessian(negative_loglik)(self._map_experiment._params, X, Y)
        precision = jnp.diag(precision['w_loc']['w_loc'])
        # precision = optax.second_order.hessian_diag(negative_loglik, self._map_experiment._params, X, Y)
        # precision = precision[1:]
        precision += self._shrink
        scale = 1. / jnp.sqrt(precision)
        if numpyro.util.not_jax_tracer(scale):
            if np.any(np.isnan(scale)):
                warnings.warn(
                    "Hessian of log posterior at the MAP point is singular. Posterior"
                    " samples from AutoLaplaceApproxmiation will be constant (equal to"
                    " the MAP point). Please consider using an AutoNormal guide.",
                    stacklevel=numpyro.util.find_stack_level(),
                )
        print(f"Laplace mean % = {jnp.isnan(scale).mean()}")
        scale = jnp.where(jnp.isnan(scale), 0.0, scale)
        loc = self._map_experiment._params["w_loc"]
        self._posterior = dist.Normal(loc, scale).to_event(1)

    @property
    def posterior(self) -> tuple[dist.Distribution, Optional[dist.Distribution]]:
        return self._posterior, None

    def make_predictions(self, rng_key_predict: random.PRNGKey):
        assert self._map_experiment._params is not None and self._map_experiment._guide is not None
        assert self._posterior is not None
        X_test, _ = self._data.test
        samples = self._posterior.sample(rng_key_predict, sample_shape=(self._num_samples,))
        predictive = Predictive(model=self._bnn, posterior_samples={'w': samples})
        self._predictions = predictive(rng_key_predict, X=X_test, Y=None)  # ['Y'][..., 0]

    @staticmethod
    def from_pickle(bnn: BayesianNeuralNetwork, data: Data, filename: str):
        experiment = Experiment.from_pickle(bnn, data, filename)
        experiment._map_experiment._bnn = bnn
        experiment._map_experiment._data = data
        experiment._map_experiment._svi = None
        experiment._map_experiment._saved_svi_state = None
        experiment._map_experiment._guide = None
        experiment._map_experiment._lr_schedule = optax.constant_schedule(-0.005)  # Can't be pickled
        # Make sure state is initialised
        experiment._map_experiment.train(random.PRNGKey(0), num_iter=0)
        return experiment


class AutoFullRankLaplaceExperiment2(AutoDiagonalLaplaceExperiment):
    def train(self, rng_key_train: random.PRNGKey):
        loss = TraceMeanField_ELBO(num_particles=1)
        rng_key = random.PRNGKey(0)

        def negative_loglik(params, inputs, targets):
            return loss.loss(rng_key, params, self._bnn, self._map_experiment._guide, X=inputs, Y=targets)

        X, Y = self._data.train
        # precision = optax.second_order.fisher_diag(negative_loglik, self._map_experiment._params, X, Y)
        precision = jax.hessian(negative_loglik)(self._map_experiment._params, X, Y)
        precision = precision['w_loc']['w_loc']
        precision += self._shrink*jnp.eye(precision.shape[0])
        cov = jnp.linalg.inv(precision)
        if numpyro.util.not_jax_tracer(cov):
            if np.any(np.isnan(cov)):
                warnings.warn(
                    "Hessian of log posterior at the MAP point is singular. Posterior"
                    " samples from AutoLaplaceApproxmiation will be constant (equal to"
                    " the MAP point). Please consider using an AutoNormal guide.",
                    stacklevel=numpyro.util.find_stack_level(),
                )
        print(f"Laplace mean % = {jnp.isnan(cov).mean().mean()}")
        cov = jnp.where(jnp.isnan(cov), 0.0, cov)
        loc = self._map_experiment._params["w_loc"]
        self._posterior = dist.MultivariateNormal(loc, cov)

    @property
    def posterior(self) -> tuple[dist.Distribution, Optional[dist.Distribution]]:
        return self._posterior, self._bnn_post._prior_prec_obs


class SWAGExperiment(SequentialExperimentBlock):
    def __init__(self, bnn: BayesianNeuralNetwork, data: Data, trained_map_experiment: AutoDeltaVIExperiment,
                 rank: int = 1, learning_rate: float = 0.05, max_iter: int = 5_000, freq: int = 100,
                 num_samples: int = 2_000):
        super().__init__(bnn, data)
        self._map_experiment = trained_map_experiment
        self.rank = rank  # if rank == 1 we use diagonal approximation
        if learning_rate > 0:
            learning_rate = -learning_rate
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.freq = freq
        self.num_samples = num_samples
        # Initialise state
        self._posterior: Optional[dist.Distribution] = None

    def train(self, rng_key_train: random.PRNGKey, eps: float = 1e-30):
        swag_transform = optax_swag.swag_diag(self.freq) if self.rank < 2 else optax_swag.swag(self.freq, self.rank)
        swag_optim = optax.chain(optax.clip_by_global_norm(10.0),
                                 optax.scale(self.learning_rate),
                                 swag_transform)
        svi = SVI(self._bnn, self._map_experiment._guide, swag_optim, Trace_ELBO())
        assert hasattr(self._map_experiment, "_params"), "MAP experiment is not trained"
        swag_run_results = svi.run(
            rng_key_train, num_steps=self.max_iter, stable_update=True, init_params=self._map_experiment._params,
            X=self._data.train[0], Y=self._data.train[1])
        swag_state = swag_run_results.state.optim_state[1][1][-1]
        swag_loc = swag_state.mean['w_loc']
        swag_scale = jnp.sqrt(jnp.clip(swag_state.params2['w_loc'] - jnp.square(swag_loc), a_min=eps))
        if self.rank < 2:
            # Diagonal SWAG
            self._posterior = dist.Normal(swag_loc, swag_scale).to_event(1)
        else:
            swag_cov_diag = jnp.square(swag_scale) / 2.
            swag_cov_factor = swag_state.dparams['w_loc'].T / jnp.sqrt(2 * (self.rank - 1))
            self._posterior = dist.LowRankMultivariateNormal(swag_loc, swag_cov_factor, swag_cov_diag)

    def make_predictions(self, rng_key_predict: random.PRNGKey):
        assert self._posterior is not None
        X_test, _ = self._data.test
        rng_key_sample_w, rng_key_sample_Y = random.split(rng_key_predict)
        posterior_samples = {'w': self._posterior.sample(rng_key_sample_w, sample_shape=(self.num_samples,))}
        predictive = Predictive(model=self._bnn, posterior_samples=posterior_samples)
        self._predictions = predictive(rng_key_sample_Y, X=X_test, Y=None)

    @property
    def posterior(self) -> tuple[dist.Distribution, Optional[dist.Distribution]]:
        return self._posterior, None

    @staticmethod
    def from_pickle(bnn: BayesianNeuralNetwork, data: Data, filename: str):
        experiment = Experiment.from_pickle(bnn, data, filename)
        experiment._map_experiment._bnn = bnn
        experiment._map_experiment._data = data
        experiment._map_experiment._svi = None
        experiment._map_experiment._saved_svi_state = None
        experiment._map_experiment._guide = None
        experiment._map_experiment._lr_schedule = optax.constant_schedule(-0.005)  # Can't be pickled
        # Make sure state is initialised
        experiment._map_experiment.train(random.PRNGKey(0), num_iter=0)
        return experiment


def plot_prior_samples(bnn: BayesianNeuralNetwork, data: Data, ndraws=15, nsamples=400, fig=None, ax=None, legend=False):
    if fig is None or ax is None:
        fig, ax = plt.subplots()
    t = data.test[0][:, 1]

    with handlers.seed(rng_seed=random.PRNGKey(0)):
        # for _ in range(nsamples):
        #     prior_fn = handlers.trace(bnn).get_trace(X=data.test[0], Y=None)
        #     mu = prior_fn['Y_mean']['value'].squeeze()
        for idraw in range(ndraws):
            prior_fn = handlers.trace(bnn).get_trace(X=data.test[0], Y=None)
            mu = prior_fn['Y_mean']['value'].squeeze()
            sigma = prior_fn['Y_scale']['value'].squeeze() if 'Y_scale' in prior_fn.keys() else \
                prior_fn['sigma_obs']['value']
            mu_label = "Posterior function draws" if idraw == 0 else None
            ax.plot(t, mu, label=mu_label)
            std_label = "+- stds of observation" if idraw == 0 else None
            ax.fill_between(t, mu - sigma, mu + sigma, alpha=0.15, label=std_label)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    if legend:
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        from matplotlib.legend_handler import HandlerTuple, HandlerLine2D
        legend_elements = [(
            (
                Line2D([0], [0], color='blue', lw=1),
                Patch(facecolor='blue', lw=0, alpha=0.1)
            ),
            (
                Line2D([0], [0], color='orange', lw=1),
                Patch(facecolor='orange', lw=0, alpha=0.1)
            ),
            Line2D([0], [0], color='black', lw=0, marker='o', markerfacecolor='black', markersize=0.5),
        )]
        handler_map = {
            legend_elements[0]: HandlerTuple(ndivide=3, pad=None),
            Line2D: HandlerLine2D(marker_pad=0.05, numpoints=3)
        }
        leg = ax.legend(legend_elements, ['Function draws (mean+-std)'], handler_map=handler_map)
        leg._legend_elements_arg = legend_elements

    return fig


class SequentialExperiment(SequentialExperimentBlock):
    def __init__(self, bnn: BayesianNeuralNetwork, data: Data, Block: Type[SequentialExperimentBlock],
                 num_inference_steps: int = 2, **block_kwargs):
        """ :param num_inference_steps: split data into this many chunks, and
                                        do Bayesian inference sequentially on them
        """
        super().__init__(bnn, data)
        self._num_inference_steps = num_inference_steps
        self._Block: Type[SequentialExperimentBlock] = Block
        self._block_kwargs: dict = block_kwargs
        # Initialise state
        self._experiment_blocks: list[Block] = list()

    @property
    def posterior(self) -> tuple[dist.Distribution, Optional[dist.Distribution]]:
        assert len(self._experiment_blocks) > 0
        return self._experiment_blocks[-1].posterior

    def train(self, rng_key_train: random.PRNGKey):
        train_len = self._data.train[0].shape[0]
        rng_key_array_train: random.PRNGKeyArray = random.split(rng_key_train, num=self._num_inference_steps)
        for step_idx, rng_key_train_step in enumerate(rng_key_array_train):
            chunk = slice(step_idx * (train_len // self._num_inference_steps),
                          min(train_len, (step_idx + 1) * (train_len // self._num_inference_steps)))
            data_view = DataSlice(self._data, chunk)
            experiment_block = self._Block(self._bnn, data_view, **self._block_kwargs)
            experiment_block.train(rng_key_train_step)
            self._bnn = self._bnn.with_prior(*experiment_block.posterior)
            self._experiment_blocks.append(experiment_block)

    def make_predictions(self, rng_key_predict: random.PRNGKey, final_only: bool = True):
        # Delegate to final experiment block
        assert len(self._experiment_blocks) > 0
        if final_only:
            self._experiment_blocks[-1].make_predictions(rng_key_predict)
        else:
            rng_key_array: random.PRNGKeyArray = random.split(rng_key_predict, len(self._experiment_blocks))
            for experiment_block, rng_key in zip(self._experiment_blocks, rng_key_array):
                experiment_block.make_predictions(rng_key)

    def make_plots(self, final_only: bool = True, **kwargs) -> plt.Figure:
        # fig, ax = plt.subplots(nrows=len(self._experiment_blocks))
        assert len(self._experiment_blocks) > 0
        if final_only:
            return self._experiment_blocks[-1].make_plots()
        else:
            for experiment_block in self._experiment_blocks:
                experiment_block.make_plots()
        return None


class ExperimentWithLastBlockReplaced(Experiment):
    def __init__(self, sequential_experiment: SequentialExperiment, LastBlock: Type[Experiment], **kwargs):
        super().__init__(sequential_experiment._bnn, sequential_experiment._data)
        self._LastBlock: Type[Experiment] = LastBlock
        self._sequential_experiment: SequentialExperiment = sequential_experiment
        self._kwargs = kwargs
        self._last_block: Optional[Experiment] = None

    def train(self, rng_key_train: random.PRNGKey):
        rng_seq, rng_hmc = random.split(rng_key_train)
        self._sequential_experiment.train(rng_seq)
        last_seq_block = self._sequential_experiment._experiment_blocks[-1]
        last_block = self._LastBlock(bnn=last_seq_block._bnn, data=last_seq_block._data, **self._kwargs)
        last_block.train(rng_hmc)
        self._last_block = last_block
        # self._sequential_experiment._experiment_blocks[-1] = last_block

    def make_predictions(self, rng_key_predict: random.PRNGKey, **kwargs):
        rng_seq, rng_hmc = random.split(rng_key_predict)
        self._sequential_experiment.make_predictions(rng_seq, **kwargs)
        self._last_block.make_predictions(rng_hmc)

    def make_plots(self, final_only: bool = True, **kwargs) -> plt.Figure:
        self._sequential_experiment.make_plots(final_only)
        self._last_block.make_plots(**kwargs)
