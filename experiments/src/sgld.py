# Copyright 2020 The Google Research Authors.
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modifications Copyright 2025 Gabor Pituk.
"""Optax and NumPyro implementations of SGMCMC optimizers."""

from collections import namedtuple
from typing import Any, NamedTuple

import jax
from jax import device_put, lax, random, vmap
import jax.numpy as jnp
from numpyro.infer.mcmc import MCMCKernel
from numpyro.infer.util import (
    ParamInfo, init_to_uniform, initialize_model)
from numpyro.util import identity, is_prng_key
from optax import GradientTransformation, Params, apply_updates


def _normal_like_tree(a, key):
    treedef = jax.tree.structure(a)
    num_vars = len(jax.tree.leaves(a))
    all_keys = jax.random.split(key, num=num_vars+1)
    noise = jax.tree.map(lambda p, k: jax.random.normal(k, shape=p.shape), a,
                            jax.tree.unflatten(treedef, all_keys[1:]))
    return noise, all_keys[0]


Momentum = Any  # An arbitrary pytree of `jnp.ndarrays`
GradMomentEstimates = Params  # Same type as parameters
PreconditionerState = NamedTuple  # State of a preconditioner


class OptaxSGLDState(NamedTuple):
    """Optax state for the SGLD optimizer"""
    count: jnp.ndarray
    rng_key: jnp.ndarray
    momentum: Momentum
    preconditioner_state: PreconditionerState


def sgld_gradient_update(step_size_fn, momentum_decay=0., preconditioner=None):
    """Optax implementation of the SGLD optimizer.

    If momentum_decay is set to zero, we get the SGLD method [1]. Otherwise,
    we get the underdamped SGLD (SGHMC) method [2].

    Args:
    step_size_fn: a function taking training step as input and producing the
        step size as output.
    rng_key: `jax.random.PRNGKey', random key.
    momentum_decay: float, momentum decay parameter (default: 0).
    preconditioner: Preconditioner, an object representing the preconditioner
        or None; if None, identity preconditioner is used (default: None).  [1]
        "Bayesian Learning via Stochastic Gradient Langevin Dynamics" Max
        Welling, Yee Whye Teh; ICML 2011  [2] "Stochastic Gradient Hamiltonian
        Monte Carlo" Tianqi Chen, Emily B. Fox, Carlos Guestrin; ICML 2014
    """

    if preconditioner is None:
        preconditioner = get_identity_preconditioner()

    def init_fn(params, rng_key):
        return OptaxSGLDState(
            count=jnp.zeros([], jnp.int32),
            rng_key=rng_key,
            momentum=jax.tree.map(jnp.zeros_like, params),
            preconditioner_state=preconditioner.init(params))

    def update_fn(gradient, state, params=None):
        del params
        lr = step_size_fn(state.count)
        lr_sqrt = jnp.sqrt(lr)
        noise_std = jnp.sqrt(2 * (1 - momentum_decay))

        preconditioner_state = preconditioner.update_preconditioner(
            gradient, state.preconditioner_state)

        noise, new_key = _normal_like_tree(gradient, state.rng_key)
        noise = preconditioner.multiply_by_m_sqrt(noise, preconditioner_state)

        def update_momentum(m, g, n):
            return momentum_decay * m + g * lr_sqrt + n * noise_std

        momentum = jax.tree.map(update_momentum, state.momentum, gradient, noise)
        updates = preconditioner.multiply_by_m_inv(momentum, preconditioner_state)
        updates = jax.tree.map(lambda m: m * lr_sqrt, updates)
        return updates, OptaxSGLDState(
            count=state.count + 1,
            rng_key=new_key,
            momentum=momentum,
            preconditioner_state=preconditioner_state)

    return GradientTransformation(init_fn, update_fn)


class Preconditioner(NamedTuple):
    """Preconditioner transformation"""
    init: Any  # TODO @izmailovpavel: fix
    update_preconditioner: Any
    multiply_by_m_sqrt: Any
    multiply_by_m_inv: Any
    multiply_by_m_sqrt_inv: Any


class IdentityPreconditionerState(PreconditionerState):
    """Identity preconditioner is stateless."""


def get_identity_preconditioner():
    """ Identity preconditioning """
    def init_fn(_):
        return IdentityPreconditionerState()

    def update_preconditioner_fn(*_, **__):
        return IdentityPreconditionerState()

    def multiply_by_m_inv_fn(vec, _):
        return vec

    def multiply_by_m_sqrt_fn(vec, _):
        return vec

    def multiply_by_m_sqrt_inv_fn(vec, _):
        return vec

    return Preconditioner(
        init=init_fn,
        update_preconditioner=update_preconditioner_fn,
        multiply_by_m_inv=multiply_by_m_inv_fn,
        multiply_by_m_sqrt=multiply_by_m_sqrt_fn,
        multiply_by_m_sqrt_inv=multiply_by_m_sqrt_inv_fn)


IntegratorState = namedtuple("IntegratorState", ["u", "optax_state"])
IntegratorState.__new__.__defaults__ = (None,) * len(IntegratorState._fields)


def euler_integrator(
    potential_fn,
    step_size_fn,
    momentum_decay=0.,
    preconditioner=None
):
    r"""
    Euler integrator scheme to sample the Langevin dynamics.
    :param potential_fn: Python callable that computes the potential, i.e.,
        negative log posterior to sample from.
    :param step_size_fn: Python callable that gives a step size schedule
        given the number of warmup steps.
    :param float momentum_decay: whether to include momentum for under-dampened
        Langevin dynamics. Defaults to 0 for standard SGLD.
    :param preconditioner: Python callable that preconditions the Euler update.
    """
    sgld_init, sgld_update = sgld_gradient_update(
        step_size_fn, momentum_decay, preconditioner)

    def init_fn(rng_key, init_params):
        # Initialize Optax state with initial parameters
        optax_state = sgld_init(init_params, rng_key)
        return IntegratorState(init_params, optax_state)

    def update_fn(state, forward_mode_differentiation=False):
        u, optax_state = state

        # Compute gradient of potential (negative log posterior)
        if forward_mode_differentiation:
            grad_potential = jax.jacfwd(potential_fn)(u)
        else:
            grad_potential = jax.grad(potential_fn)(u)
        grad_log_posterior = jax.tree.map(lambda x: -x, grad_potential)  # ∇ log π

        # Apply Optax SGLD update
        updates, new_optax_state = sgld_update(
            grad_log_posterior, optax_state, None
        )
        u_new = apply_updates(u, updates)

        return IntegratorState(u_new, new_optax_state)

    return init_fn, update_fn


# Define the SGLD state (integrator state)
SGLDState = namedtuple(
    "SGLDState", ["u", "euler_state", "momentum_decay", "num_warmup"])


def sgld(
    potential_fn=None,
    potential_fn_gen=None,
    preconditioner=None,
    step_size_fn=None,
):
    r"""
    Stochastic Gradient Langevin Dynamics inference, with optional
    preconditioner.

    **References:**

    1. *Bayesian Learning via Stochastic Gradient Langevin Dynamics*
       Max Welling, Yee Whye Teh; ICML 2011
    2. *Stochastic Gradient Hamiltonian Monte Carlo*
       Tianqi Chen, Emily B. Fox, Carlos Guestrin; ICML 2014

    :param potential_fn: Python callable that computes the potential energy
        given input parameters. The input parameters to `potential_fn` can be
        any python collection type, provided that `init_params` argument to
        `init_kernel` has the same type.
    :param potential_fn_gen: Python callable that when provided with model
        arguments / keyword arguments returns `potential_fn`. This
        may be provided to do inference on the same model with changing data.
        If the data shape remains the same, we can compile `sample_kernel`
        once, and use the same for multiple inference runs.
    :param preconditioner: Python callable that preconditions the Euler update.
    :param step_size_fn: Python callable that gives a step size schedule
        given the number of warmup steps.
    :return: a tuple of callables (`init_kernel`, `sample_kernel`), the first
        one to initialize the sampler, and the second one to generate samples
        given an existing one.

    .. warning::
        Instead of using this interface directly, we would highly recommend you
        to use the higher level :class:`~numpyro.infer.mcmc.MCMC` API instead.

    **Example**

    .. doctest::

        >>> import jax
        >>> from jax import random
        >>> import jax.numpy as jnp
        >>> import numpyro
        >>> import numpyro.distributions as dist
        >>> from numpyro.infer.sgld import sgld
        >>> from numpyro.infer.util import initialize_model
        >>> from numpyro.util import fori_collect
        >>> true_coefs = jnp.array([1., 2., 3.])
        >>> data = random.normal(random.PRNGKey(2), (2000, 3))
        >>> labels = dist.Bernoulli(logits=(true_coefs * data).sum(-1)).sample(random.PRNGKey(3))
        >>>
        >>> def model(data, labels):
        ...      coefs = numpyro.sample('coefs', dist.Normal(jnp.zeros(3), jnp.ones(3)))
        ...      intercept = numpyro.sample('intercept', dist.Normal(0., 10.))
        ...      return numpyro.sample('y', dist.Bernoulli(logits=(coefs * data + intercept).sum(-1)), obs=labels)
        >>>
        >>> model_info = initialize_model(random.PRNGKey(0), model, model_args=(data, labels,))
        >>> init_kernel, sample_kernel = sgld(model_info.potential_fn,
        >>>                                   step_size_fn=lambda num_warmup: lambda step: 0.0001)
        >>> sgld_state = init_kernel(model_info.param_info,
        ...                         num_warmup=4000)
        >>> samples = fori_collect(0, 10_000, sample_kernel, sgld_state,
        ...                        transform=lambda state: model_info.postprocess_fn(state.u))
        >>> print(jnp.mean(samples['coefs'], axis=0))  # doctest: +SKIP
        [0.912652  2.0404327 2.8986464]
    """
    sgld_update = None
    forward_mode_ad = False

    def init_kernel(
        init_params,
        num_warmup,
        momentum_decay=0.,
        forward_mode_differentiation=False,
        model_args=(),
        model_kwargs=None,
        rng_key=None,
    ):
        """
        Initializes the SGLD sampler.

        :param init_params: Initial parameters to begin sampling. The type must
            be consistent with the input type to `potential_fn`.
        :param int num_warmup: Number of warmup steps; samples generated
            during warmup are discarded.
        :param float momentum_decay: whether to include momentum for under-dampened
            Langevin dynamics. Defaults to 0 for standard SGLD.
        :param bool forward_mode_differentiation: whether to use forward-mode differentiation
            or reverse-mode differentiation. By default, we use reverse mode but the forward
            mode can be useful in some cases to improve the performance. In addition, some
            control flow utility on JAX such as `jax.lax.while_loop` or `jax.lax.fori_loop`
            only supports forward-mode differentiation. See
            `JAX's The Autodiff Cookbook <https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html>`_
            for more information.
        :param tuple model_args: Model arguments if `potential_fn_gen` is specified.
        :param dict model_kwargs: Model keyword arguments if `potential_fn_gen` is specified.
        :param jax.random.PRNGKey rng_key: random key to be used as the source of
            randomness.

        """
        nonlocal forward_mode_ad
        forward_mode_ad = forward_mode_differentiation
        rng_key = random.PRNGKey(0) if rng_key is None else rng_key
        if isinstance(init_params, ParamInfo):
            (init_params,
             _,  # pe,
             __,  # z_grad
            ) = init_params
        pe_fn = potential_fn
        if potential_fn_gen:
            if pe_fn is not None:
                raise ValueError(
                    "Only one of `potential_fn` or `potential_fn_gen` must be provided."
                )
            kwargs = {} if model_kwargs is None else model_kwargs
            pe_fn = potential_fn_gen(*model_args, **kwargs)

        # Configure Optax SGLD
        nonlocal sgld_update
        sgld_init, sgld_update = euler_integrator(
            pe_fn, step_size_fn(num_warmup), momentum_decay, preconditioner)
        init_euler_state = sgld_init(rng_key, init_params)
        sgld_state = SGLDState(
            init_params,
            init_euler_state,
            momentum_decay=momentum_decay,
            num_warmup=num_warmup,
            # XXX review if want to flatten state
            # vv_state.z,
            # vv_state.z_grad,
            # vv_state.potential_energy,
            # wa_state,
            # rng_key_hmc,
        )
        return device_put(sgld_state)

    def sample_kernel(sgld_state, model_args=(), model_kwargs=None):
        """
        Given an existing :data:`SGLDState`, run SGLD step and return a new
        :data:`SGLDState`.

        :param sgld_state: Current sample (and associated state).
        :param tuple model_args: Model arguments if `potential_fn_gen` is specified.
        :param dict model_kwargs: Model keyword arguments if `potential_fn_gen` is specified.
        :return: new :data:`SGLDState` from simulating Langevin dynamics given existing
        state.

        """
        model_kwargs = {} if model_kwargs is None else model_kwargs

        nonlocal sgld_update, forward_mode_ad
        if potential_fn_gen:
            pe_fn = potential_fn_gen(*model_args, **model_kwargs)
            _, sgld_update = euler_integrator(
                pe_fn, step_size_fn(sgld_state.num_warmup),
                sgld_state.momentum_decay, preconditioner)

        new_state = sgld_update(sgld_state.euler_state, forward_mode_ad)
        # jax.debug.print("{pre} -> {post}", pre=sgld_state.euler_state.u, post=new_state.u)

        return SGLDState(
            new_state.u,
            new_state,
            momentum_decay=sgld_state.momentum_decay,
            num_warmup=sgld_state.num_warmup,
        )

    return init_kernel, sample_kernel


class SGLD(MCMCKernel):
    """
    Stochastic Gradient Langevin Dynamics inference, using unadjusted
    Langevin Dynamics to draw approximate samples from a distribution,
    allowing provision for step size and preconditioning (Stochastic
    Gradient Hamiltonian Monte Carlo).

    **References:**

    1. *Bayesian Learning via Stochastic Gradient Langevin Dynamics*
       Max Welling, Yee Whye Teh; ICML 2011
    2. *Stochastic Gradient Hamiltonian Monte Carlo*
       Tianqi Chen, Emily B. Fox, Carlos Guestrin; ICML 2014

    .. note:: Until the kernel is used in an MCMC run, `postprocess_fn` will return the
        identity function.
    """
    def __init__(
        self,
        model=None,
        potential_fn=None,
        step_size=None,
        step_size_fn=None,
        init_strategy=init_to_uniform,
        momentum_decay=0.,
        preconditioner=None,
        forward_mode_differentiation=False,

    ):
        """
        :param model: Python callable containing Pyro :mod:`~numpyro.primitives`.
            If model is provided, `potential_fn` will be inferred using the model.
        :param potential_fn: Python callable that computes the potential energy
            given input parameters. The input parameters to `potential_fn` can be
            any python collection type, provided that `init_params` argument to
            :meth:`init` has the same type.
        :param float step_size: Determines the size of a single step taken by the
            Euler integrator while computing the trajectory using Langevin
            dynamics. One of this, or `step_size_fn` needs to be specified.
        :param callable step_size_fn: a learning rate schedule generator
            function. Given an int num_warmup, produces a learning rate schedule
            mapping the iteration index to a learning rate.
        :param callable init_strategy: a per-site initialization function.
            See :ref:`init_strategy` section for available functions.
        :param bool forward_mode_differentiation: whether to use forward-mode differentiation
            or reverse-mode differentiation. By default, we use reverse mode but the forward
            mode can be useful in some cases to improve the performance. In addition, some
            control flow utility on JAX such as `jax.lax.while_loop` or `jax.lax.fori_loop`
            only supports forward-mode differentiation. See
            `JAX's The Autodiff Cookbook <https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html>`_
            for more information.
        """
        if not (model is None) ^ (potential_fn is None):
            raise ValueError("Only one of `model` or `potential_fn` must be specified.")
        if not (step_size is None) ^ (step_size_fn is None):
            raise ValueError("Only one of `step_size` or `step_size_fn` must be specified.")
        if step_size_fn is None:
            step_size = lax.convert_element_type(step_size, jnp.result_type(float))
            def step_size_fn(_):  # num_warmup
                def schedule(_):  # step
                    return step_size
                return schedule
        self._model = model
        self._potential_fn = potential_fn
        self._step_size_fn = step_size_fn
        self._init_strategy = init_strategy
        self._momentum_decay = momentum_decay
        self._preconditioner = preconditioner
        self._forward_mode_differentiation = forward_mode_differentiation
        # Set on first call to init
        self._optim = None
        self._init_fn = None
        self._potential_fn_gen = None
        self._postprocess_fn = None
        self._sample_fn = None

    def _init_state(self, rng_key, model_args, model_kwargs, init_params):
        if self._model is not None:
            (
                new_init_params,
                potential_fn,
                postprocess_fn,
                _,  # model_trace
            ) = initialize_model(
                rng_key,
                self._model,
                dynamic_args=True,
                init_strategy=self._init_strategy,
                model_args=model_args,
                model_kwargs=model_kwargs,
                forward_mode_differentiation=self._forward_mode_differentiation,
            )
            if init_params is None:
                init_params = new_init_params
            if self._init_fn is None:
                self._init_fn, self._sample_fn = sgld(
                    potential_fn_gen=potential_fn,
                    preconditioner=self._preconditioner,
                    step_size_fn=self._step_size_fn,
                )
            self._potential_fn_gen = potential_fn
            self._postprocess_fn = postprocess_fn
        elif self._init_fn is None:
            self._init_fn, self._sample_fn = sgld(
                potential_fn=self._potential_fn,
                preconditioner=self._preconditioner,
                step_size_fn=self._step_size_fn,
            )

        return init_params

    @property
    def model(self):
        """ Access the numpyro model if provided. """
        return self._model

    @property
    def sample_field(self):
        return "u"

    @property
    def default_fields(self):
        return ("u",)

    def get_diagnostics_str(self, state):
        count = state.euler_state.optax_state.count
        return f"{count} steps of size " \
               f"{self._step_size_fn(state.num_warmup)(count):.2e}."

    def init(self, rng_key, num_warmup, init_params=None, model_args=(),
             model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = dict()
        # non-vectorized
        if is_prng_key(rng_key):
            rng_key, rng_key_init_model = random.split(rng_key)
        # vectorized
        else:
            rng_key, rng_key_init_model = jnp.swapaxes(
                vmap(random.split)(rng_key), 0, 1
            )
        init_params = self._init_state(
            rng_key_init_model, model_args, model_kwargs, init_params
        )
        if self._potential_fn and init_params is None:
            raise ValueError("Valid value of `init_params` must be provided with"
                             " `potential_fn`.")

        if self._model is not None and isinstance(init_params, ParamInfo):
            init_params = init_params[0]

        def sgld_init_fn(init_params, rng_key):
            return self._init_fn(
                init_params,
                num_warmup=num_warmup,
                momentum_decay=self._momentum_decay,
                model_args=model_args,
                model_kwargs=model_kwargs,
                rng_key=rng_key,
            )
        if is_prng_key(rng_key):
            init_state = sgld_init_fn(init_params, rng_key)
        else:
            # XXX it is safe to run sgld_init_fn under vmap despite that sgld_init_fn changes some
            # nonlocal variables: sgld_update, forward_mode_ad, because those variables do not
            # depend on traced args: init_params, rng_key.
            init_state = vmap(sgld_init_fn)(init_params, rng_key)
            sample_fn = vmap(self._sample_fn, in_axes=(0, None, None))
            self._sample_fn = sample_fn
        return init_state

    def postprocess_fn(self, model_args, model_kwargs):
        if self._postprocess_fn is None:
            return identity
        return self._postprocess_fn(*model_args, **model_kwargs)

    def sample(self, state, model_args, model_kwargs):
        """
        Run SGLD from the given :data:`SGLDState` and return the resulting
        :data:`SGLDState`.

        :param SGLDState state: Represents the current state.
        :param model_args: Arguments provided to the model.
        :param model_kwargs: Keyword arguments provided to the model.
        :return: Next `state` after running SGLD update.
        """
        return self._sample_fn(state, model_args, model_kwargs)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_sample_fn"] = None
        state["_init_fn"] = None
        return state
