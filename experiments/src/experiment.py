__all__ = [
    "BasicHMCExperiment",
    "AutoDeltaVIExperiment",
    "AutoMeanFieldNormalVIExperiment",
    "BasicMeanFieldGaussianVIExperiment",
    "BasicFullRankGaussianVIExperiment",
    "AutoFullRankLaplaceExperiment",
    "AutoDiagonalLaplaceExperiment",
    "SWAGExperiment",
    "SequentialExperiment",
    "ExperimentWithLastBlockReplaced",
    "init_loc_fn",
    "plot_prior_samples",
]

import os
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

from .data import Data, DataSlice
from .model import BayesianNeuralNetwork


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

    def make_plots(self, fig=None, ax=None, plot_bald=False, legend=False, **kwargs) -> plt.Figure:
        assert self._predictions is not None
        X, Y = self._data.train
        X_test, _ = self._data.test
        # plotting
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        if self._bnn.OBS_MODEL != "classification":
            # compute mean prediction and confidence interval around median
            Y_mean_pred, Y_pred = self._predictions["Y_mean"][..., 0], self._predictions["Y"][..., 0]
            mean_means = jnp.mean(Y_mean_pred, axis=0)
            mean_percentiles = np.percentile(Y_mean_pred, [5.0, 95.0], axis=0)
            Y_percentiles = np.percentile(Y_pred, [5.0, 95.0], axis=0)
            # plot training data
            ax.plot(X[:, 1], Y[:, 0], "kx")
            # plot predictions & quantiles
            ax.plot(X_test[:, 1], mean_means, color="blue", label="Posterior mean")
            ax.fill_between(X_test[:, 1], *mean_percentiles, color="orange", alpha=0.5, label="90% CI on mean")
            ax.fill_between(X_test[:, 1], *Y_percentiles, color="lightgreen", alpha=0.5, label="90% prediction")
            if plot_bald:
                bald_scores = self.compute_test_bald_scores()
                ax.plot(X_test[:, 1], bald_scores, color="black", alpha=0.6, label="BALD score")
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

    # TODO: in general we can't compute this integral yeah? but eg constant prec normal obs & normal w-post. we can
    # TODO: o/w we resort to sampling posterior and logexpsum approx?
    # def test_loglik(self) -> float:
    #     raise NotImplementedError()


class BasicHMCExperiment(Experiment):
    def __init__(self, bnn: BayesianNeuralNetwork, data: Data, init_params: Optional[dict] = None,
                 num_samples: int = 2_000, num_warmup: int = 1_000, num_chains: int = 1, group_by_chain: bool = False):
        super().__init__(bnn, data)
        self._init_params = init_params
        self._num_samples = num_samples
        self._num_warmup = num_warmup
        self._num_chains = num_chains
        self._group_by_chain = group_by_chain
        # Initialise state
        self._mcmc = None
        samples_init_shape = (0, self._bnn.get_weight_dim(),) if not self._group_by_chain else \
            (num_chains, 0, self._bnn.get_weight_dim(),)
        self._samples: dict = dict(w=jnp.empty(samples_init_shape))

    def train(self, rng_key_train: random.PRNGKey, progress_bar: bool = True):
        start = time.time()
        X, Y = self._data.train
        if self._mcmc is None:
            kernel = NUTS(self._bnn)
            self._mcmc = MCMC(
                kernel,
                num_warmup=self._num_warmup,
                num_samples=self._num_samples,
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
                            raise NotImplementedError()
                        #     elbo_particle = (
                        #             elbo_particle
                        #             + numpyro.infer.elbo._get_log_prob_sum(model_site)
                        #             - numpyro.infer.elbo._get_log_prob_sum(guide_site)
                        #     )

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
            self._saved_svi_state = self._svi.init(rng_key_train, X=X, Y=Y)

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
        assert self._params is not None and self._guide is not None
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


class BasicMeanFieldGaussianVIExperiment(BasicVIExperiment):
    def _get_guide(self) -> Callable[[jnp.ndarray, Optional[jnp.ndarray]], Any]:
        bnn_weight_dim = self._bnn.get_weight_dim()

        def guide(X, Y=None):
            w_loc = numpyro.param("w_loc", lambda rng_key: dist.Normal(scale=0.25).sample(rng_key, (bnn_weight_dim,)))
            w_scale = numpyro.param("w_scale", jnp.full((bnn_weight_dim,), 1e-5),
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


class AutoMeanFieldNormalVIExperiment(BasicVIExperiment):
    def _get_guide(self) -> Callable:
        return autoguide.AutoNormal(self._bnn, init_loc_fn=numpyro.infer.init_to_sample, init_scale=1e-5)

    @property
    def posterior(self) -> tuple[dist.Distribution, dist.Distribution]:
        raise NotImplementedError()


class AutoDeltaVIExperiment(BasicVIExperiment):
    def __init__(self, bnn: BayesianNeuralNetwork, data: Data, max_iter: int = 150_000,
                 lr_schedule: optax.Schedule = optax.constant_schedule(-0.001), num_samples: int = 200):
        # Things are deterministic -> single sample enough for loss
        super().__init__(bnn, data, num_samples=num_samples, max_iter=max_iter, lr_schedule=lr_schedule,
                         num_particles=1, num_eval_particles=1)

    def _get_guide(self) -> Callable:
        bnn_weight_dim = self._bnn.get_weight_dim()
        _, prec_obs_prior = self._bnn.prior

        def guide(X, Y=None):
            w_loc = numpyro.param("w_loc", lambda rng_key: dist.Normal(scale=0.25).sample(rng_key, (bnn_weight_dim,)))
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

        def guide(X, Y=None):
            w_loc = numpyro.param("w_loc", lambda rng_key: dist.Normal().sample(rng_key, (bnn_weight_dim,)))
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
            hessian_fn=lambda f, x: jax.hessian(f)(x) + jnp.eye(x.shape[-1]) * self._shrink
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
        predictive = Predictive(model=self._bnn, posterior_samples={'w': samples})
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
        precision = optax.fisher_diag(negative_loglik, self._map_experiment._params, X, Y)
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


def plot_prior_samples(bnn: BayesianNeuralNetwork, data: Data, ndraws=15, nsamples=400, fig=None, ax=None):
    if fig is None or ax is None:
        fig, ax = plt.subplots()
    t = data.test[0][:, 1]

    with handlers.seed(rng_seed=random.PRNGKey(0)):
        # for _ in range(nsamples):
        #     prior_fn = handlers.trace(bnn).get_trace(X=data.test[0], Y=None)
        #     mu = prior_fn['Y_mean']['value'].squeeze()
        for _ in range(ndraws):
            prior_fn = handlers.trace(bnn).get_trace(X=data.test[0], Y=None)
            mu = prior_fn['Y_mean']['value'].squeeze()
            sigma = prior_fn['Y_scale']['value'].squeeze() if 'Y_scale' in prior_fn.keys() else \
                prior_fn['sigma_obs']['value']
            ax.plot(t, mu)
            ax.fill_between(t, mu - sigma, mu + sigma, alpha=0.05)
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
