# This file contains default factories, setting appropriate tuned hyperparameters for different algorithms
__all__ = [
    "small",
    "big",
    "small_inv_gamma",
    "big_inv_gamma",
    "small_fixed_prec",
    "big_fixed_prec",
]
import jax
import jax.numpy as jnp
import numpy as np
import optax
from typing import List

from jax import random

from .model import BayesianLinearRegression, BNNRegressor
from .data import Data
from .experiment import *


class factory:
    D_X = 2
    BETA = 1.0
    HMC_NUM_CHAINS = 4

    @classmethod
    def bnn(cls) -> BNNRegressor:
        raise NotImplementedError()

    @classmethod
    def map(cls, bnn: BNNRegressor, data: Data) -> AutoDeltaVIExperiment:
        raise NotImplementedError()

    @classmethod
    def hmc(cls, bnn: BNNRegressor, data: Data, trained_map_experiments: List[AutoDeltaVIExperiment]) -> BasicHMCExperiment:
        raise NotImplementedError()

    @classmethod
    def map_then_hmc(cls, bnn: BNNRegressor, data: Data) -> BasicHMCExperiment:
        def train_delta(rng_key):
            delta = cls.map(bnn, data)
            delta.train(rng_key)
            return delta
        deltas = [train_delta(rng_key) for rng_key in random.split(random.PRNGKey(0), num=cls.HMC_NUM_CHAINS)]
        return cls.hmc(bnn, data, deltas)

    @classmethod
    def sgld(cls, bnn: BNNRegressor, data: Data, trained_map_experiments: list[AutoDeltaVIExperiment], step_size=0.0001) -> BasicSGLDExperiment:
        raise NotImplementedError()

    @classmethod
    def map_then_sgld(cls, bnn: BNNRegressor, data: Data) -> BasicHMCExperiment:
        def train_delta(rng_key):
            delta = cls.map(bnn, data)
            delta.train(rng_key)
            return delta
        deltas = [train_delta(rng_key) for rng_key in random.split(random.PRNGKey(0), num=cls.HMC_NUM_CHAINS)]
        return cls.sgld(bnn, data, deltas)

    @classmethod
    def mfvi(cls, bnn: BNNRegressor, data: Data) -> BasicMeanFieldGaussianVIExperiment:
        raise NotImplementedError()

    @classmethod
    def diag_laplace(cls, bnn: BNNRegressor, data: Data, trained_map_experiment: AutoDeltaVIExperiment) -> AutoDiagonalLaplaceExperiment:
        raise NotImplementedError()

    @classmethod
    def map_then_diag_laplace(cls, bnn: BNNRegressor, data: Data) -> AutoDiagonalLaplaceExperiment:
        print(cls.__name__)
        delta = cls.map(bnn, data)
        delta.train(random.PRNGKey(0))
        return cls.diag_laplace(bnn, data, delta)

    @classmethod
    def swag(cls, bnn: BNNRegressor, data: Data, trained_map_experiment: AutoDeltaVIExperiment) -> SWAGExperiment:
        raise NotImplementedError()

    @classmethod
    def map_then_swag(cls, bnn: BNNRegressor, data: Data) -> SWAGExperiment:
        delta = cls.map(bnn, data)
        delta.train(random.PRNGKey(0))
        return cls.swag(bnn, data, delta)


class small(factory):
    @classmethod
    def bnn(cls) -> BNNRegressor:
        return BNNRegressor(
            nonlin=jax.nn.silu,
            D_X=cls.D_X,
            D_Y=1,
            D_H=[32, 32, 16],
            biases=True,
            obs_model="loc_scale",
            prior_scale=np.sqrt(2),
            prior_type="xavier",
            beta=1.0,
            scale_nonlin=lambda xs: jax.nn.softplus(xs) * 0.2 + 1e-2,
        )

    @classmethod
    def map(cls, bnn: BNNRegressor, data: Data) -> AutoDeltaVIExperiment:
        return AutoDeltaVIExperiment(
            bnn, data, max_iter=100_000,
            lr_schedule=optax.piecewise_constant_schedule(
                -0.005, {50_000: 0.5}
            ),
        )

    @classmethod
    def hmc(cls, bnn: BNNRegressor, data: Data, trained_map_experiments: list[AutoDeltaVIExperiment]) -> BasicHMCExperiment:
        assert len(trained_map_experiments) == cls.HMC_NUM_CHAINS
        init_params = jnp.array([delta._params["w_loc"] for delta in trained_map_experiments])
        return BasicHMCExperiment(
            bnn, data,
            init_params={"w": init_params},
            num_warmup=50, num_samples=400,
            num_chains=cls.HMC_NUM_CHAINS
        )

    @classmethod
    def map_then_hmc(cls, bnn: BNNRegressor, data: Data) -> BasicHMCExperiment:
        return super().map_then_hmc(bnn, data)

    @classmethod
    def sgld(cls, bnn: BNNRegressor, data: Data, trained_map_experiments: list[AutoDeltaVIExperiment], step_size=1e-5) -> BasicSGLDExperiment:
        assert len(trained_map_experiments) == cls.HMC_NUM_CHAINS
        init_params = jnp.array([delta._params["w_loc"] for delta in trained_map_experiments])
        return BasicSGLDExperiment(
            bnn, data,
            init_params={"w": init_params},
            init_step_size=2*step_size,
            final_step_size=step_size,
            thinning=250,
            num_warmup=50_000,
            num_samples=100_000,
            num_chains=cls.HMC_NUM_CHAINS,
        )

    @classmethod
    def diag_laplace(cls, bnn: BNNRegressor, data: Data, trained_map_experiment: AutoDeltaVIExperiment) -> AutoDiagonalLaplaceExperiment:
        return AutoDiagonalLaplaceExperiment(bnn, data, trained_map_experiment,
                                             shrink=300, num_samples=400)

    @classmethod
    def map_then_diag_laplace(cls, bnn: BNNRegressor, data: Data) -> AutoDiagonalLaplaceExperiment:
        return super().map_then_diag_laplace(bnn, data)

    @classmethod
    def swag(cls, bnn: BNNRegressor, data: Data, trained_map_experiment: AutoDeltaVIExperiment) -> SWAGExperiment:
        return SWAGExperiment(bnn, data, trained_map_experiment,
                              # rank=1,
                              rank=20,
                              learning_rate=0.02,
                              max_iter=10_000,
                              freq=25,  # 400 usable samples to fit Gaussian
                              num_samples=400)

    @classmethod
    def map_then_swag(cls, bnn: BNNRegressor, data: Data) -> SWAGExperiment:
        return super().map_then_swag(bnn, data)

    BETA = 0.325

    @classmethod
    def mfvi(cls, bnn: BNNRegressor, data: Data) -> BasicMeanFieldGaussianVIExperiment:
        return BasicMeanFieldGaussianVIExperiment(
            bnn, data, num_samples=400,
            max_iter=75_000,
            lr_schedule=optax.piecewise_constant_schedule(-0.001, {25_000: 0.5, 50_000: 0.5}),
            num_particles=16, num_eval_particles=128,
        )


class big(factory):
    @classmethod
    def bnn(cls) -> BNNRegressor:
        return BNNRegressor(
            nonlin=jax.nn.silu,
            D_X=cls.D_X,
            D_Y=1,
            D_H=[128, 256, 128, 64],
            biases=True,
            obs_model="loc_scale",
            prior_scale=np.sqrt(2),
            prior_type="xavier",
            beta=1.0,
            scale_nonlin=lambda xs: jax.nn.softplus(xs) * 0.2 + 1e-2
        )

    @classmethod
    def map(cls, bnn: BNNRegressor, data: Data) -> AutoDeltaVIExperiment:
        # orig_beta = bnn.BETA
        # bnn.BETA = 0.
        # pretrain_map = AutoDeltaVIExperiment(bnn, data, max_iter=5_000)
        return AutoDeltaVIExperiment(
            bnn, data, max_iter=50_000,
            lr_schedule=optax.piecewise_constant_schedule(
                -0.002, {12_500: 0.5}
            ),
        )

    @classmethod
    def hmc(cls, bnn: BNNRegressor, data: Data, trained_map_experiments: list[AutoDeltaVIExperiment]) -> BasicHMCExperiment:
        assert len(trained_map_experiments) == cls.HMC_NUM_CHAINS
        init_params = jnp.array([delta._params["w_loc"] for delta in trained_map_experiments])
        return BasicHMCExperiment(
            bnn, data,
            init_params={"w": init_params},
            num_warmup=50, num_samples=150,
            num_chains=cls.HMC_NUM_CHAINS
        )

    @classmethod
    def map_then_hmc(cls, bnn: BNNRegressor, data: Data) -> BasicHMCExperiment:
        return super().map_then_hmc(bnn, data)

    @classmethod
    def sgld(cls, bnn: BNNRegressor, data: Data, trained_map_experiments: list[AutoDeltaVIExperiment], step_size=5e-7) -> BasicSGLDExperiment:
        assert len(trained_map_experiments) == cls.HMC_NUM_CHAINS
        init_params = jnp.array([delta._params["w_loc"] for delta in trained_map_experiments])
        return BasicSGLDExperiment(
            bnn, data,
            init_params={"w": init_params},
            init_step_size=2*step_size,
            final_step_size=step_size,
            thinning=250,
            num_warmup=50_000,
            num_samples=100_000,
            num_chains=cls.HMC_NUM_CHAINS,
        )

    @classmethod
    def diag_laplace(cls, bnn: BNNRegressor, data: Data, trained_map_experiment: AutoDeltaVIExperiment) -> AutoDiagonalLaplaceExperiment:
        return AutoDiagonalLaplaceExperiment(bnn, data, trained_map_experiment,
                                             shrink=2_000, num_samples=400)

    @classmethod
    def map_then_diag_laplace(cls, bnn: BNNRegressor, data: Data) -> AutoDiagonalLaplaceExperiment:
        return super().map_then_diag_laplace(bnn, data)

    @classmethod
    def swag(cls, bnn: BNNRegressor, data: Data, trained_map_experiment: AutoDeltaVIExperiment) -> SWAGExperiment:
        return SWAGExperiment(bnn, data, trained_map_experiment,
                              # rank=1,
                              rank=20,
                              learning_rate=0.01,
                              max_iter=4_000,
                              freq=10,  # 400 usable samples to fit Gaussian
                              num_samples=400)

    @classmethod
    def map_then_swag(cls, bnn: BNNRegressor, data: Data) -> SWAGExperiment:
        return super().map_then_swag(bnn, data)

    BETA = 0.05

    @classmethod
    def mfvi(cls, bnn: BNNRegressor, data: Data) -> BasicMeanFieldGaussianVIExperiment:
        return BasicMeanFieldGaussianVIExperiment(
            bnn, data, num_samples=400,
            max_iter=40_000,
            lr_schedule=optax.piecewise_constant_schedule(-0.001, {25_000: 0.5, 50_000: 0.5}),
            num_particles=1, num_eval_particles=16,
        )


class inv_gamma_mixin:
    """Mixin to configure BNN with inverse Gamma observation model."""
    @classmethod
    def hmc(cls, bnn: BNNRegressor, data: Data, trained_map_experiments: list[AutoDeltaVIExperiment]) -> BasicHMCExperiment:
        assert len(trained_map_experiments) == cls.HMC_NUM_CHAINS
        init_params = jnp.array([delta._params["w_loc"] for delta in trained_map_experiments])
        init_prec_obs = jnp.full((len(trained_map_experiments),), bnn.prior[1].mean)
        return BasicHMCExperiment(
            bnn, data,
            init_params={"w": init_params, "prec_obs": init_prec_obs},
            num_warmup=50, num_samples=150,
            num_chains=cls.HMC_NUM_CHAINS
        )

    @classmethod
    def sgld(cls, bnn: BNNRegressor, data: Data, trained_map_experiments: list[AutoDeltaVIExperiment], step_size=5e-7) -> BasicSGLDExperiment:
        assert len(trained_map_experiments) == cls.HMC_NUM_CHAINS
        init_params = jnp.array([delta._params["w_loc"] for delta in trained_map_experiments])
        init_prec_obs = jnp.full((len(trained_map_experiments),), bnn.prior[1].mean)
        return BasicSGLDExperiment(
            bnn, data,
            init_params={"w": init_params, "prec_obs": init_prec_obs},
            init_step_size=2*step_size,
            final_step_size=step_size,
            thinning=250,
            num_warmup=50_000,
            num_samples=100_000,
            num_chains=cls.HMC_NUM_CHAINS,
        )


    @classmethod
    def _override_bnn(cls, original_bnn: BNNRegressor) -> BNNRegressor:
        return BNNRegressor(
            nonlin=original_bnn._nonlin,
            D_X=original_bnn.D_X,
            D_Y=original_bnn.D_Y-1,
            D_H=original_bnn.D_H,
            biases=original_bnn._biases,
            obs_model=("inv_gamma", 10., 0.0225),  # High confidence
            prior_scale=np.sqrt(2),
            prior_type = "xavier",
            beta=original_bnn.BETA,
            scale_nonlin=original_bnn._scale_nonlin,
        )


class small_inv_gamma(inv_gamma_mixin, small):
    @classmethod
    def bnn(cls) -> BNNRegressor:
        return cls._override_bnn(super().bnn())

    @classmethod
    def mfvi(cls, bnn: BNNRegressor, data: Data) -> InvGammaObservedMeanFieldGaussianVIExperiment:
        return InvGammaObservedMeanFieldGaussianVIExperiment(
            bnn, data, num_samples=400,
            max_iter=75_000,
            lr_schedule=optax.piecewise_constant_schedule(-0.001, {25_000: 0.5, 50_000: 0.5}),
            num_particles=16, num_eval_particles=128,
        )


class big_inv_gamma(inv_gamma_mixin, big):
    @classmethod
    def bnn(cls) -> BNNRegressor:
        return cls._override_bnn(super().bnn())

    @classmethod
    def mfvi(cls, bnn: BNNRegressor, data: Data) -> InvGammaObservedMeanFieldGaussianVIExperiment:
        return InvGammaObservedMeanFieldGaussianVIExperiment(
            bnn, data, num_samples=400,
            max_iter=40_000,
            lr_schedule=optax.piecewise_constant_schedule(-0.001, {25_000: 0.5, 50_000: 0.5}),
            num_particles=1, num_eval_particles=16,
        )


class small_fixed_prec(small):
    @classmethod
    def bnn(cls) -> BNNRegressor:
        original_bnn = super().bnn()
        return BNNRegressor(
            nonlin=original_bnn._nonlin,
            D_X=original_bnn.D_X,
            D_Y=original_bnn.D_Y-1,
            D_H=original_bnn.D_H,
            biases=original_bnn._biases,
            obs_model=25.,  # Override to const precision
            prior_scale=np.sqrt(2),
            prior_type="xavier",
            beta=1.0,
            scale_nonlin=lambda xs: jax.nn.softplus(xs) * 0.2 + 1e-2,
        )


class big_fixed_prec(big):
    @classmethod
    def bnn(cls) -> BNNRegressor:
        original_bnn = super().bnn()
        return BNNRegressor(
            nonlin=original_bnn._nonlin,
            D_X=original_bnn.D_X,
            D_Y=original_bnn.D_Y-1,
            D_H=original_bnn.D_H,
            biases=original_bnn._biases,
            obs_model=25.,  # Override to const precision
            prior_scale=np.sqrt(2),
            prior_type="xavier",
            beta=1.0,
            scale_nonlin=lambda xs: jax.nn.softplus(xs) * 0.2 + 1e-2,
        )

    @classmethod
    def hmc(cls, bnn: BNNRegressor, data: Data, trained_map_experiments: list[AutoDeltaVIExperiment]) -> BasicHMCExperiment:
        assert len(trained_map_experiments) == cls.HMC_NUM_CHAINS
        init_params = jnp.array([delta._params["w_loc"] for delta in trained_map_experiments])
        return BasicHMCExperiment(
            bnn, data,
            init_params={"w": init_params},
            num_warmup=10, num_samples=800,
            num_chains=cls.HMC_NUM_CHAINS
        )


class tractable(factory):
    @classmethod
    def bnn(cls) -> BayesianLinearRegression:
        return BayesianLinearRegression(
            input_dim=20,
            noise_scale=0.15,
            prior_scale=1,
        )

    @classmethod
    def true(cls, blr: BayesianLinearRegression, data: Data) -> TrueBayesianLinearRegressionExperiment:
        return TrueBayesianLinearRegressionExperiment(blr, data)

    BETA = 1.0

    @classmethod
    def mfvi(cls, blr: BayesianLinearRegression, data: Data) -> BasicMeanFieldGaussianVIExperiment:
        return BasicMeanFieldGaussianVIExperiment(
            blr, data, num_samples=400,
            max_iter=200_000,
            lr_schedule=optax.piecewise_constant_schedule(-0.0001, {40_000: 0.5, 80_000: 0.5}),
            num_particles=64, num_eval_particles=1,
            init_scale=1e-2,
        )

    @classmethod
    def fullrank_vi(cls, blr: BayesianLinearRegression, data: Data) -> BasicFullRankGaussianVIExperiment:
        return BasicFullRankGaussianVIExperiment(
            blr, data, num_samples=400,
            max_iter=400_000,
            lr_schedule=optax.piecewise_constant_schedule(-0.0001, {75_000: 0.5, 150_000: 0.5}),
            num_particles=1, num_eval_particles=1,
        )

    @classmethod
    def map(cls, blr: BayesianLinearRegression, data: Data) -> AutoDeltaVIExperiment:
        return AutoDeltaVIExperiment(
            blr, data, max_iter=150_000,
            lr_schedule=optax.piecewise_constant_schedule(
                -0.0001, {50_000: 0.5}
            ),
        )

    @classmethod
    def hmc(cls, bnn: BNNRegressor, data: Data, trained_map_experiments: list[AutoDeltaVIExperiment]) -> BasicHMCExperiment:
        assert len(trained_map_experiments) == cls.HMC_NUM_CHAINS
        init_params = jnp.array([delta._params["w_loc"] for delta in trained_map_experiments])
        return BasicHMCExperiment(
            bnn, data,
            init_params={"w": init_params},
            num_warmup=50, num_samples=400,
            num_chains=cls.HMC_NUM_CHAINS
        )

    @classmethod
    def map_then_hmc(cls, bnn: BNNRegressor, data: Data) -> BasicHMCExperiment:
        return super().map_then_hmc(bnn, data)

    @classmethod
    def sgld(cls, blr: BNNRegressor, data: Data, trained_map_experiments: list[AutoDeltaVIExperiment], step_size=1e-5) -> BasicSGLDExperiment:
        assert len(trained_map_experiments) == cls.HMC_NUM_CHAINS
        init_params = jnp.array([delta._params["w_loc"] for delta in trained_map_experiments])
        return BasicSGLDExperiment(
            blr, data,
            init_params={"w": init_params},
            init_step_size=2*1e-6,
            final_step_size=8*1e-7,
            thinning=250,
            num_warmup=50_000,
            num_samples=250_000,
            num_chains=cls.HMC_NUM_CHAINS,
        )

    @classmethod
    def diag_laplace(cls, blr: BayesianLinearRegression, data: Data, trained_map_experiment: AutoDeltaVIExperiment) -> AutoDiagonalLaplaceExperiment:
        return AutoDiagonalLaplaceExperiment(
            blr, data,
            trained_map_experiment=trained_map_experiment,
            # num_samples=400,
            # max_iter=40_000,
            # lr_schedule=optax.piecewise_constant_schedule(-0.0001, {75_000: 0.5, 150_000: 0.5}),
            # num_particles=16, num_eval_particles=1,
            shrink=0.,
        )

    @classmethod
    def map_then_diag_laplace(cls, bnn: BNNRegressor, data: Data) -> AutoDiagonalLaplaceExperiment:
        return super().map_then_diag_laplace(bnn, data)

    @classmethod
    def fullrank_laplace(cls, blr: BayesianLinearRegression, data: Data, trained_map_experiment: AutoDeltaVIExperiment) -> AutoFullRankLaplaceExperiment2:
        return AutoFullRankLaplaceExperiment2(
            blr, data,
            trained_map_experiment=trained_map_experiment,
            # num_samples=400,
            # max_iter=40_000,
            # lr_schedule=optax.piecewise_constant_schedule(-0.0001, {75_000: 0.5, 150_000: 0.5}),
            # num_particles=16, num_eval_particles=1,
            shrink=0,
        )

    @classmethod
    def map_then_fullrank_laplace(cls, bnn: BNNRegressor, data: Data) -> AutoDiagonalLaplaceExperiment:
        print(cls.__name__)
        delta = cls.map(bnn, data)
        delta.train(random.PRNGKey(0))
        return cls.fullrank_laplace(bnn, data, delta)

    @classmethod
    def swag(cls, bnn: BNNRegressor, data: Data, trained_map_experiment: AutoDeltaVIExperiment) -> SWAGExperiment:
        raise NotImplementedError

    @classmethod
    def map_then_swag(cls, bnn: BNNRegressor, data: Data) -> SWAGExperiment:
        raise NotImplementedError
