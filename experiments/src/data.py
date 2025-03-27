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
    def __init__(self, gen_D_X: int = 3, feat_D_X = 2, sigma_obs: float = 0.05, train_size: int = 50, test_size: int = 500):
        assert feat_D_X >= 2
        self.gen_D_X = gen_D_X
        self.feat_D_X = feat_D_X
        self.sigma_obs = sigma_obs
        D_Y = 1  # create 1d outputs
        np.random.seed(0)
        self.W = 0.5 * np.random.randn(gen_D_X)
        X = jnp.concatenate((jnp.linspace(-1, -0.4, train_size // 2),
                             jnp.linspace(0.4, 1, train_size - (train_size // 2))))
        X = self._feature_expand(X)
        Y = self._get_unscaled_mean(X)
        Y += sigma_obs * np.random.randn(train_size, D_Y)
        self._sub_mean = jnp.mean(Y)
        self._div_std = jnp.std(Y)
        Y = self._scale(Y)

        assert X.shape == (train_size, feat_D_X)
        assert Y.shape == (train_size, D_Y)

        X_test = jnp.linspace(-3., 3., test_size)
        X_test = self._feature_expand(X_test)

        self._X = X
        self._Y = Y
        self._X_test = X_test
        self._Y_test = None

    def _feature_expand(self, X: jax.Array, length_scale=1.0):
        if self.feat_D_X == 2:
            # Hack so that this input dimension has no effect on anything
            # we rely on true x being second coordinate
            return jnp.hstack((jnp.zeros((X.shape[0], 1,)), X[:, np.newaxis],))
        np.random.seed(0)
        omega = np.random.randn(self.feat_D_X) / length_scale
        b = np.random.uniform(low=0., high=2.*np.pi, size=self.feat_D_X)
        expanded_X = jnp.cos(jnp.add(jnp.outer(X, omega), b)) * jnp.sqrt(2.)
        expanded_X = expanded_X.at[:, 1].set(X)
        return expanded_X

    def _get_unscaled_mean(self, X: jax.Array):
        # X should be feature-expanded already
        # y = w0 + w1*x + w2*x**2 + 1/2 (1/2+x)**2 * sin(4x)
        t = X[:, 1]
        underlying_X = jnp.power(t[:, np.newaxis], jnp.arange(self.gen_D_X))
        Y = jnp.dot(underlying_X, self.W) + 0.5 * jnp.power(0.5 + t, 2.0) * jnp.sin(4.0 * t)
        # Y = jnp.dot(X[:, :(min(3, self.D_X))], self.W) + 0.5 * jnp.power(0.5 + X[:, 1], 2.0) * jnp.sin(4.0 * X[:, 1])
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


def FlatData():
    return DataSlice(ToyData1(gen_D_X=2, train_size=100), slice(0, 50))


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
    def __init__(self, intercept: float, beta: float, D_X=2, sigma_obs=0.05, train_size=50, test_size=500):
        self.intercept = intercept
        self.beta = beta
        assert D_X >= 2
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

    def _feature_expand(self, X: jax.Array, length_scale=1.0):
        if self.D_X == 2:
            # Hack so that this input dimension has no effect on anything
            # we rely on true x being second coordinate
            return jnp.hstack((jnp.zeros((X.shape[0], 1,)), X[:, np.newaxis],))
        np.random.seed(0)
        omega = np.random.randn(self.D_X) / length_scale
        b = np.random.uniform(low=0., high=2.*np.pi, size=self.D_X)
        expanded_X = jnp.cos(jnp.add(jnp.outer(X, omega), b)) * jnp.sqrt(2.)
        expanded_X = expanded_X.at[:, 1].set(X)
        return expanded_X

    def _get_unscaled_mean(self, X: jax.Array):
        # X should be feature-expanded already
        # y = icpt + b*x
        Y = self.intercept + self.beta * X[:, 1]
        # Y = jnp.dot(X[:, :2], jnp.array([self.intercept, self.beta]))
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
        # X = jnp.concatenate(Xs, axis=0)
        # Y = None if any(Yi is None for Yi in Ys) else jnp.concatenate(Ys, axis=0)
        return Xs[0], Ys[0]

    def true_predictive(self, X: jax.Array) -> dist.Distribution:
        raise NotImplementedError()


class GPData(Data):
    def __init__(self, length_scale: float = 1.0,
             sigma_obs: float = 0.1,
             train_size: int = 50,
             test_size: int = 500,
             rff_full_features: int = 500,
             rff_features: int = 20):
        # Kernel parameters
        self.length_scale = length_scale
        self.sigma_obs = sigma_obs
        self.rff_full_features = rff_full_features
        self.rff_features = rff_features

        THOLD =2
        # Generate train points in [-3, -THOLD) and [THOLD, 3)
        train_part1 = jnp.linspace(-3, -THOLD, num=train_size//2, endpoint=False)
        train_part2 = jnp.linspace(THOLD, 3, num=train_size - train_size//2, endpoint=False)
        raw_train_x = jnp.concatenate([train_part1, train_part2])

        # Generate test points in [-6, -3), [-THOLD, THOLD), and [3, 6]
        test_part_size = test_size // 3
        remainder = test_size % 3
        test_part1 = jnp.linspace(-6, -3, num=test_part_size + (remainder > 0), endpoint=False)
        test_part2 = jnp.linspace(-THOLD, THOLD, num=test_part_size + (remainder > 1), endpoint=False)
        test_part3 = jnp.linspace(3, 6, num=test_part_size, endpoint=False)
        raw_test_x = jnp.concatenate([test_part1, train_part1, test_part2, train_part2, test_part3])
        train_mask = jnp.concatenate([jnp.zeros_like(test_part1), jnp.ones_like(train_part1), jnp.zeros_like(test_part2),
                                      jnp.ones_like(train_part2), jnp.zeros_like(test_part3)]).astype(bool)

        # Combine all points for GP sampling
        X_full = raw_test_x
        K = self._se_kernel(X_full, X_full)
        self._K = K

        # Sample function values from GP prior
        np.random.seed(0)
        f_full = jnp.array(np.random.multivariate_normal(mean=np.zeros(K.shape[0]), cov=K))

        # Split into train/test
        self._f_train = f_full[train_mask]
        self._f_test = f_full

        # Add noise only to train observations
        noise = np.random.normal(0, sigma_obs, size=train_size)
        self._Y_train = self._f_train + noise
        self._Y_train = self._Y_train[:, np.newaxis]
        self._Y_test = self._f_test  # No noise for test
        self._Y_test = self._Y_test[:, np.newaxis]

        # Initialize RFF parameters
        self._omega = np.random.normal(0, 1/length_scale, rff_full_features)
        self._b = np.random.uniform(0, 2*np.pi, rff_full_features)

        # Feature-expand inputs
        self._center_X = None
        self._X_train = self._feature_expand(raw_train_x)
        self._X_test = self._feature_expand(raw_test_x)

    def _se_kernel(self, x1: jax.Array, x2: jax.Array) -> jax.Array:
        """Squared exponential kernel matrix"""
        dists = jnp.subtract.outer(x1, x2)
        return jnp.exp(-0.5 * (dists ** 2) / (self.length_scale ** 2))

    def _feature_expand(self, X: jax.Array) -> jax.Array:
        """RFF expansion matching the SE kernel"""
        # Use precomputed omega and b
        scaled_inputs = jnp.outer(X, self._omega)
        phases = scaled_inputs + self._b
        features = jnp.cos(phases) * jnp.sqrt(2.0 / self.rff_features)
        def get_whitening_matrix(X, epsilon=1e-5):
            import numpy as np
            """
            Whitens a matrix using ZCA whitening.

            Args:
                X (np.ndarray): Input data matrix of shape (n_samples, n_features).
                epsilon (float): Small value to avoid division by zero.

            Returns:
                np.ndarray: Whitened data matrix.
            """
            # 1. Mean-center the data
            X_centered = X - np.mean(X, axis=0)
            centering = np.mean(X, axis=0)

            # 2. Compute covariance matrix
            cov = np.cov(X_centered, rowvar=False)

            # 3. Eigendecomposition
            eigenvalues, eigenvectors = np.linalg.eigh(cov)

            # 4. Construct whitening matrix
            whitening_matrix = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues + epsilon)) @ eigenvectors.T

            # 5. Whiten the data
            return centering, whitening_matrix
        def select_uncorrelated_subset(cov_matrix, subset_size):
            import numpy as onp
            """
            Selects a subset of dimensions with minimal maximum absolute covariance.

            Args:
                cov_matrix (np.ndarray): A 100x100 covariance matrix.
                subset_size (int): The desired size of the subset.

            Returns:
                list: Indices of the selected subset, ordered by their selection sequence.
            """
            n = cov_matrix.shape[0]
            A = onp.abs(cov_matrix.copy())
            onp.fill_diagonal(A, 0)  # Ignore the diagonal elements (variances)

            selected = []
            remaining = list(range(n))

            for _ in range(subset_size):
                if not selected:
                    # First selection: variable with the smallest sum of absolute covariances
                    sums = A.sum(axis=1)
                    min_idx = onp.argmin(sums)
                else:
                    # Submatrix of remaining variables vs selected variables
                    submatrix = A[onp.ix_(remaining, selected)]
                    # Find maximum covariance with any selected variable for each remaining
                    max_covs = submatrix.max(axis=1)
                    # Select the variable with the smallest maximum covariance
                    min_idx_pos = onp.argmin(max_covs)
                    min_idx = remaining[min_idx_pos]
                selected.append(min_idx)
                remaining.remove(min_idx)

            return onp.array(selected)
        if self._center_X is None:
            self._center_X, self._whiten_X = get_whitening_matrix(features, epsilon=1e-5)
        # if self._subidx is None:
        #     self._subidx = jnp.concatenate([
        #         jnp.array([0, 1]),
        #         2+select_uncorrelated_subset(features[:, 2:].T @ features[:, 2:], subset_size=self.rff_features),
        #     ])
        # features = features.at[:, 0].set(1.)
        # features = (features - self._center_X) @ self._whiten_X
        features = features.at[:, 1].set(X)
        # features = features[:, self._subidx]
        return features

    @property
    def train(self) -> tuple[jax.Array, jax.Array]:
        return self._X_train, self._Y_train

    @property
    def test(self) -> tuple[jax.Array, jax.Array]:
        return self._X_test, self._Y_test

    def true_predictive(self, X: jax.Array) -> dist.Distribution:
        """Returns the true noise distribution around the function"""
        raise NotImplementedError()
        # Feature-expand inputs
        X_expanded = self._feature_expand(X)

        # Compute true function values (would need to recompute covariance)
        # For simplicity, we'll return the observation distribution
        return dist.Normal(loc=0, scale=self.sigma_obs).expand([X.shape[0]])
