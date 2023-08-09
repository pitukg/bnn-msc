import copy
from abc import ABC, abstractmethod
from typing import Callable, Optional

import jax
import jax.nn
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro import handlers
from typing_extensions import Self


class BayesianNeuralNetwork(ABC):
    def __init__(self, beta: float = 1.0):
        self.OBS_MODEL = None
        self.BETA = beta

    @abstractmethod
    def __call__(self, X: jax.Array, Y: Optional[jax.Array] = None) -> None:
        raise NotImplementedError()

    @abstractmethod
    def get_weight_dim(self) -> int:
        raise NotImplementedError()

    @property
    @abstractmethod
    def prior(self) -> tuple[dist.Distribution, Optional[dist.Distribution]]:
        raise NotImplementedError()

    @abstractmethod
    def with_prior(self, prior_w: dist.Distribution) -> Self:
        raise NotImplementedError()


class BNNRegressor(BayesianNeuralNetwork):
    def __init__(self, nonlin: Callable[[jax.Array], jax.Array], D_X: int, D_Y: int, D_H: list[int], biases: bool,
                 obs_model: str | float = "loc_scale", prior_scale: float = 1.0, prior_type: str = "iid",
                 beta: float = 1.0):
        """ :param obs_model: float: precision of Gaussian / "loc_scale": predict both / "inv_gamma": Gamma
            hyper-prior on precision / "classification" for softmax classifier on D_Y classes
        """
        super().__init__(beta)
        self._nonlin = nonlin
        # map scales into R+ using softplus ie log(1+exp(.))
        self._scale_nonlin = lambda xs: jax.nn.softplus(xs) + 1e-2  # Add eps so lik doesn't vanish
        self.D_X = D_X
        self.D_Y = D_Y
        self.D_H = D_H
        self._biases = biases
        if obs_model == "loc_scale":
            if self.D_Y > 1:
                raise NotImplementedError("Should predict a cov matrix... not impl yet")
            self.OBS_MODEL = "loc_scale"
            self.D_Y += 1
            assert self.D_Y == 2
        elif obs_model == "inv_gamma":
            self.OBS_MODEL = "inv_gamma"
            # self._prior_prec_obs = dist.Gamma(3.0, 1.0)
            self._prior_prec_obs = dist.Gamma(1.0, 0.025)
        elif obs_model == "classification":
            self.OBS_MODEL = "classification"
        elif isinstance(obs_model, float):
            self.OBS_MODEL = "const_prec"
            # Abstract const parameter into dist; mask according to convention below, see guides
            self._prior_prec_obs = dist.Delta(obs_model).mask(False)
        # add trainable numpyro.param too?
        else:
            raise ValueError(obs_model)

        assert prior_type in ("iid", "xavier")
        prior_scales = self._scale_init(prior_scale, prior_type)
        # Initialise priors to independent standard normals
        self._prior_w = dist.Normal(jnp.zeros(self.get_weight_dim()), prior_scales).to_event(1)
        # self._prior_w = dist.MultivariateNormal(jnp.zeros(self.get_weight_dim()),
        #                                         jnp.diag(jnp.full((self.get_weight_dim(),), prior_scale)))

    def get_weight_dim(self) -> int:
        if self._biases:
            dim = 0
            prev = self.D_X
            for width in self.D_H:
                dim += prev * width + width
                prev = width
            dim += prev * self.D_Y + self.D_Y
            return dim
        else:
            dim = 0
            prev = self.D_X
            for width in self.D_H:
                dim += prev * width
                prev = width
            dim += prev * self.D_Y
            return dim

    def _wi_from_flat(self, a: jax.Array, depth: int, bias: bool = False) -> jax.Array:
        # set bias to return bias of that layer
        assert a.shape[0] == self.get_weight_dim()
        assert 0 <= depth <= len(self.D_H)
        if bias:
            assert self._biases
        prev = self.D_X
        idx = 0
        layer = 0
        for width in self.D_H:
            if depth == layer:
                if not bias:
                    return a[idx:(idx + prev * width)].reshape((prev, width))
                else:
                    idx += prev * width
                    return a[idx:(idx + width)]  # .reshape((width, 1))
            idx += prev * width
            if self._biases:
                idx += width
            layer += 1
            prev = width
        assert depth == layer == len(self.D_H)
        if not bias:
            return a[idx:(idx + prev * self.D_Y)].reshape((prev, self.D_Y))
        else:
            idx += prev * self.D_Y
            return a[idx:(idx + self.D_Y)]  # .reshape((self.D_Y, 1))

    def _scale_init(self, prior_scale, prior_type: str) -> jnp.array:
        res = np.full((self.get_weight_dim(),), prior_scale, dtype=float)
        if prior_type == "iid":
            return res
        assert prior_type == "xavier"
        idx = jnp.arange(self.get_weight_dim())
        for depth, width in enumerate([self.D_X] + self.D_H):
            # Divide variance by dimension of current hidden state -> divide scale by sqrt
            res[self._wi_from_flat(idx, depth)] /= np.sqrt(width)
            if self._biases:
                res[self._wi_from_flat(idx, depth, bias=True)] /= np.sqrt(width)
        return res

    # noinspection PyPep8Naming
    def __call__(self, X: jax.Array, Y: Optional[jax.Array] = None):
        N, D_X = X.shape
        assert D_X == self.D_X

        # sample weights from prior
        with handlers.scale(scale=self.BETA):
            w = numpyro.sample("w", self._prior_w)

        pre_activ = jnp.matmul(X, self._wi_from_flat(w, depth=0))
        if self._biases:
            pre_activ += self._wi_from_flat(w, depth=0, bias=True)
        for depth in range(1, 1 + len(self.D_H)):
            pre_activ = jnp.matmul(self._nonlin(pre_activ), self._wi_from_flat(w, depth))
            if self._biases:
                pre_activ += self._wi_from_flat(w, depth, bias=True)

        if self.OBS_MODEL == "loc_scale":
            assert pre_activ.shape[-1] == 2
            Y_mean = numpyro.deterministic("Y_mean", pre_activ[..., [0]])
            if Y is not None:
                assert Y_mean.shape == Y.shape
            Y_scale = numpyro.deterministic("Y_scale", self._scale_nonlin(pre_activ[..., [1]]))
            # observe data
            with numpyro.plate("data", N):
                numpyro.sample("Y", dist.Normal(Y_mean, Y_scale).to_event(1), obs=Y)

        elif self.OBS_MODEL == "classification":
            assert pre_activ.shape[-1] >= 2
            Y_p = numpyro.deterministic("Y_p", jax.nn.softmax(pre_activ, axis=1)[..., jnp.newaxis, :])

            # observe data
            with numpyro.plate("data", N):
                numpyro.sample("Y", dist.Categorical(probs=Y_p).to_event(1), obs=Y)

        else:
            assert hasattr(self, "_prior_prec_obs")
            # we put a prior on the observation noise
            prec_obs = numpyro.sample("prec_obs", self._prior_prec_obs)
            sigma_obs = numpyro.deterministic("sigma_obs", 1.0 / jnp.sqrt(prec_obs))

            Y_mean = numpyro.deterministic("Y_mean", pre_activ)
            if Y is not None:
                assert Y_mean.shape == Y.shape

            # observe data
            with numpyro.plate("data", N):
                numpyro.sample("Y", dist.Normal(Y_mean, jnp.full((N, self.D_Y), sigma_obs)).to_event(1), obs=Y)

    @property
    def prior(self) -> tuple[dist.Distribution, Optional[dist.Distribution]]:
        """ :returns prior on w and (prec_obs if exists)"""
        return self._prior_w, self._prior_prec_obs if hasattr(self, "_prior_prec_obs") else None

    def with_prior(self, prior_w: dist.Distribution, prior_prec_obs: Optional[dist.Distribution] = None) -> Self:
        cpy = copy.deepcopy(self)
        cpy._prior_w = prior_w
        cpy._prior_prec_obs = prior_prec_obs
        return cpy
