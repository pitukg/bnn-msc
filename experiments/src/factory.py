# This file contains default factories, setting appropriate tuned hyperparameters for different algorithms
import jax
import numpy as np
import optax

from .model import BNNRegressor
from .data import Data
from .experiment import *


class factory:
    BETA = 1.0
    @staticmethod
    def bnn() -> BNNRegressor:
        raise NotImplementedError()

    @staticmethod
    def map(bnn: BNNRegressor, data: Data) -> AutoDeltaVIExperiment:
        raise NotImplementedError()

    @staticmethod
    def hmc(bnn: BNNRegressor, data: Data, trained_map_experiment: AutoDeltaVIExperiment) -> Experiment:
        raise NotImplementedError()

    @staticmethod
    def mfvi(bnn: BNNRegressor, data: Data) -> Experiment:
        raise NotImplementedError()

    @staticmethod
    def diag_laplace(bnn: BNNRegressor, data: Data, trained_map_experiment: AutoDeltaVIExperiment) -> Experiment:
        raise NotImplementedError()

    @staticmethod
    def swag(bnn: BNNRegressor, data: Data, trained_map_experiment: AutoDeltaVIExperiment) -> Experiment:
        raise NotImplementedError()


class small(factory):
    @staticmethod
    def bnn() -> BNNRegressor:
        return BNNRegressor(
            nonlin=jax.nn.silu,
            D_X=2,
            D_Y=1,
            D_H=[32, 32, 16],
            biases=True,
            obs_model="loc_scale",
            prior_scale=np.sqrt(2),
            prior_type="xavier",
            beta=1.0
        )

    @staticmethod
    def map(bnn: BNNRegressor, data: Data) -> AutoDeltaVIExperiment:
        return AutoDeltaVIExperiment(
            bnn, data, max_iter=100_000,
            lr_schedule=optax.piecewise_constant_schedule(
                -0.005, {50_000: 0.5}
            ),
        )

    @staticmethod
    def hmc(bnn: BNNRegressor, data: Data, trained_map_experiment: AutoDeltaVIExperiment) -> Experiment:
        return BasicHMCExperiment(
            bnn, data,
            init_params={"w": trained_map_experiment._params["w_loc"]},
            num_warmup=50, num_samples=400
        )

    @staticmethod
    def diag_laplace(bnn: BNNRegressor, data: Data, trained_map_experiment: AutoDeltaVIExperiment) -> Experiment:
        return AutoDiagonalLaplaceExperiment(bnn, data, trained_map_experiment,
                                             shrink=1_000, num_samples=400)

    @staticmethod
    def swag(bnn: BNNRegressor, data: Data, trained_map_experiment: AutoDeltaVIExperiment) -> Experiment:
        return SWAGExperiment(bnn, data, trained_map_experiment,
                              rank=20,
                              learning_rate=0.005,
                              max_iter=10_000,
                              freq=25,  # 400 usable samples to fit Gaussian
                              num_samples=400)

    BETA = 0.25

    @staticmethod
    def mfvi(bnn: BNNRegressor, data: Data) -> Experiment:
        return BasicMeanFieldGaussianVIExperiment(
            bnn, data, num_samples=400,
            max_iter=100_000,
            lr_schedule=optax.piecewise_constant_schedule(-0.002, {25_000: 0.25}),
            num_particles=1, num_eval_particles=16,
        )
