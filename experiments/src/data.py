import abc
from typing import Iterable

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist


class Data(abc.ABC):
    @property
    @abc.abstractmethod
    def train(self) -> tuple[jax.Array, jax.Array]:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def test(self) -> tuple[jax.Array, jax.Array]:
        raise NotImplementedError()

    # @abc.abstractmethod
    def true_predictive(self, X: jax.Array) -> dist.Distribution:
        raise NotImplementedError()


# Create partial view decorator of data
class DataSlice(Data):
    def __init__(self, data: Data, train_idx_slice: slice):
        self._data = data
        self._train_idx_slice: slice = train_idx_slice

    @property
    def train(self) -> tuple[jax.Array, jax.Array]:
        X, Y = self._data.train
        return X[self._train_idx_slice], Y[self._train_idx_slice]

    @property
    def test(self) -> tuple[jax.Array, jax.Array]:
        return self._data.test

    def true_predictive(self, X: jax.Array) -> dist.Distribution:
        return self._data.true_predictive(X)


# Reverse dataset
class ReverseData(Data):
    def __init__(self, data: Data):
        self._data = data

    @property
    def train(self) -> tuple[jax.Array, jax.Array]:
        X, Y = self._data.train
        return X[::-1, ...], Y[::-1, ...]

    @property
    def test(self) -> tuple[jax.Array, jax.Array]:
        return self._data.test

    def true_predictive(self, X: jax.Array) -> dist.Distribution:
        return self._data.true_predictive(X)


class PermutedData(Data):
    def __init__(self, data: Data, perm: np.array):
        self._data = data
        assert perm.shape[0] == data.train[0].shape[0], "wrong len"
        perm_copy = perm.copy()
        perm_copy.sort()
        assert np.all(perm_copy == np.arange(len(perm))), "not a permutation"
        self._perm = perm

    @property
    def train(self) -> tuple[jax.Array, jax.Array]:
        X, Y = self._data.train
        return X[self._perm], Y[self._perm]

    @property
    def test(self) -> tuple[jax.Array, jax.Array]:
        return self._data.test

    def true_predictive(self, X: jax.Array) -> dist.Distribution:
        return self._data.true_predictive(X)


# Define toy regression problem
# create artificial regression dataset
class ToyData1(Data):
    def __init__(self, D_X: int = 3, sigma_obs: float = 0.05, train_size: int = 50, test_size: int = 500):
        self.D_X = D_X
        self.sigma_obs = sigma_obs
        D_Y = 1  # create 1d outputs
        np.random.seed(0)
        self.W = 0.5 * np.random.randn(D_X)
        X = jnp.concatenate((jnp.linspace(-1, -0.4, train_size // 2),
                             jnp.linspace(0.4, 1, train_size - (train_size // 2))))
        X = self._feature_expand(X)
        Y = self._get_unscaled_mean(X)
        Y += sigma_obs * np.random.randn(train_size, D_Y)
        self._sub_mean = jnp.mean(Y)
        self._div_std = jnp.std(Y)
        Y = self._scale(Y)

        assert X.shape == (train_size, D_X)
        assert Y.shape == (train_size, D_Y)

        X_test = jnp.linspace(-3., 3., test_size)
        X_test = self._feature_expand(X_test)

        self._X = X
        self._Y = Y
        self._X_test = X_test
        self._Y_test = None

    def _feature_expand(self, X: jax.Array):
        return jnp.power(X[:, np.newaxis], jnp.arange(self.D_X))  # XXX ?bias included in model

    def _get_unscaled_mean(self, X: jax.Array):
        # X should be feature-expanded already
        # y = w0 + w1*x + w2*x**2 + 1/2 (1/2+x)**2 * sin(4x)
        Y = jnp.dot(X, self.W) + 0.5 * jnp.power(0.5 + X[:, 1], 2.0) * jnp.sin(4.0 * X[:, 1])
        Y = Y[:, np.newaxis]
        return Y

    def _scale(self, Y: jax.Array):
        Y -= self._sub_mean
        Y /= self._div_std
        return Y

    @property
    def train(self):
        return self._X, self._Y

    @property
    def test(self):
        return self._X_test, self._Y_test

    def true_predictive(self, X: jax.Array) -> dist.Distribution:
        X = self._feature_expand(X)
        Y_unscaled = self._get_unscaled_mean(X)
        Y = self._scale(Y_unscaled)
        scaled_sigma_obs = self.sigma_obs / self._div_std
        return dist.Normal(Y, scale=scaled_sigma_obs)


class Sign(Data):
    def __init__(self, data: Data):
        self._data = data

    @property
    def train(self) -> tuple[jax.Array, jax.Array]:
        X, y = self._data.train
        # y_sign = jnp.where(y > 0, jnp.array([[1., 0.]]), jnp.array([[0., 1.]]))
        y_sign = (y > 0).astype(jnp.int32)
        return X, y_sign

    @property
    def test(self) -> tuple[jax.Array, jax.Array]:
        X_test, y_test = self._data.test
        # y_sign = jnp.where(y_test > 0, jnp.array([[1., 0.]]), jnp.array([[0., 1.]]))
        y_sign = (y_test > 0).astype(jnp.int32) if y_test is not None else None
        return X_test, y_sign

    def true_predictive(self, X: jax.Array) -> dist.Distribution:
        underlying_pred = self._data.true_predictive(X)
        probs = 1. - underlying_pred.cdf(0.)
        return dist.Bernoulli(probs=probs)


class LinearData(Data):
    def __init__(self, intercept: float, beta: float, D_X=3, sigma_obs=0.05, train_size=50, test_size=500):
        self.intercept = intercept
        self.beta = beta
        self.D_X = D_X
        self.sigma_obs = sigma_obs
        D_Y = 1  # create 1d outputs
        np.random.seed(0)
        X = jnp.concatenate((jnp.linspace(-1, -0.4, train_size // 2),
                             jnp.linspace(0.4, 1, train_size - (train_size // 2))))
        X = self._feature_expand(X)
        Y = self._get_unscaled_mean(X)
        Y += sigma_obs * np.random.randn(train_size, D_Y)

        assert X.shape == (train_size, self.D_X)
        assert Y.shape == (train_size, D_Y)

        X_test = jnp.linspace(-3., 3., test_size)
        X_test = self._feature_expand(X_test)

        self._X = X
        self._Y = Y
        self._X_test = X_test
        self._Y_test = None

    def _feature_expand(self, X: jax.Array):
        return jnp.power(X[:, np.newaxis], jnp.arange(self.D_X))  # XXX ?bias included in model

    def _get_unscaled_mean(self, X: jax.Array):
        # X should be feature-expanded already
        # y = icpt + b*x
        Y = jnp.dot(X, jnp.array([self.intercept, self.beta]))
        Y = Y[:, np.newaxis]
        return Y

    @property
    def train(self):
        return self._X, self._Y

    @property
    def test(self):
        return self._X_test, self._Y_test

    def true_predictive(self, X: jax.Array) -> dist.Distribution:
        X = self._feature_expand(X)
        Y = self._get_unscaled_mean(X)
        return dist.Normal(Y, scale=self.sigma_obs)


class ConcatData(Data):
    def __init__(self, datas: Iterable[Data]):
        self._datas = datas

    @property
    def train(self) -> tuple[jax.Array, jax.Array]:
        trains = [data.train for data in self._datas]
        Xs = [train[0] for train in trains]
        Ys = [train[1] for train in trains]
        X = jnp.concatenate(Xs, axis=0)
        Y = jnp.concatenate(Ys, axis=0)
        return X, Y

    @property
    def test(self) -> tuple[jax.Array, jax.Array]:
        tests = [data.test for data in self._datas]
        Xs = [test[0] for test in tests]
        Ys = [test[1] for test in tests]
        X = jnp.concatenate(Xs, axis=0)
        Y = jnp.concatenate(Ys, axis=0)
        return X, Y

    def true_predictive(self, X: jax.Array) -> dist.Distribution:
        raise NotImplementedError()
