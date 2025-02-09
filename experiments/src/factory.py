# This file contains default factories, setting appropriate tuned hyperparameters for different algorithms
import jax
import jax.numpy as jnp
import numpy as np
import optax
from typing import List

from jax import random

from .model import BNNRegressor
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
