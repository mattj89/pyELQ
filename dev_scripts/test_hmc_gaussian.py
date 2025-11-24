"""

"""

import os
import numpy as np
import pandas as pd
import scipy
import time
import json
from dataclasses import dataclass
from datetime import datetime
from copy import deepcopy

import jax.numpy as jnp

import plotly.graph_objects as go
# import plotly.io as pio
# pio.renderers.default = "browser"

from openmcmc.parameter import Identity
from openmcmc.distribution.distribution import Uniform, Poisson
from openmcmc.distribution.location_scale import Normal
from openmcmc.distribution.distribution_jax import Normal_jax, Uniform_jax
from openmcmc.model import Model
from openmcmc.parameter import ScaledMatrix
from openmcmc.parameter_jax import LinearCombination_jax
from openmcmc.sampler.sampler import NormalNormal
from openmcmc.sampler.metropolis_hastings import ManifoldMALA
from openmcmc.mcmc import MCMC
from pyelq.component.source_model import HamiltonianMonteCarlo


@dataclass
class TempParameter(LinearCombination_jax):
    def update_prefactors(self, state, update_index = None):
        return state

"""
Generate a bunch of Gaussian data
"""

state = {}

real_s = 10.0
real_std = 5.0
num_obs = 100

state["y"] = jnp.atleast_2d(
    jnp.array(real_s + np.random.normal(size=num_obs) * real_std)
).T
tau = 1 / real_std ** 2

state["s"] = jnp.array([[0.0]])
state["A"] = jnp.ones(shape=(num_obs, 1))

state["B_bg"] = jnp.zeros(shape=(num_obs, 1))
state["bg"] = jnp.array([[0.0]])

state["mu"] = jnp.array([[0.0]])
prior_std = 1.0
state["lambda"] = jnp.array([[1 / prior_std ** 2]])

jit_comp_flag = True

form_dict = {"s": "A"}
source_parameter = TempParameter(
    form=form_dict,
)
state["Q"] = (tau) * scipy.sparse.eye(state["y"].size)
likelihood_y = Normal_jax(
    response="y",
    grad_list=["s"],
    mean=source_parameter,
    precision=Identity("Q"),
    terms_in_likelihood=["y", "A", "s"],
    scalar_precision=tau,
    jit_compile=jit_comp_flag
)
likelihood_y.param_list = ["s"]

log_p, state = likelihood_y.log_p(state)

# create model object
model_list = [likelihood_y]

model_list.append(
    Normal_jax(
        response="s",
        grad_list=["s"],
        mean=Identity("mu"),
        precision=Identity("lambda"),
        terms_in_likelihood=["s", "mu", "lambda"],
        scalar_precision=1 / prior_std ** 2,
        domain_response_lower=None,
        jit_compile=jit_comp_flag
    )
)
model_list[-1].param_list = ["s"]

mdl = Model(model_list)
mdl.response = {"y": "mean"}

sampler_list = []
hmc_precision = 1e0
sampler_list.append(HamiltonianMonteCarlo(
    "s", mdl, max_variable_size=(1, 1), step=0.0,
    momentum_precision=hmc_precision, epsilon=1e-1, num_leapfrog_steps=10
))

n_iter = 5000
mcmc = MCMC(state, sampler_list, model=mdl, n_burn=0, n_iter=n_iter)
mcmc.run_mcmc()

# trace plot
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=np.arange(n_iter),
        y=mcmc.store["s"].flatten(),
        mode="lines",
        name="est. param",
        line=dict(color="blue")
    )
)
fig.show()

prec_hat = state["lambda"] + num_obs * tau
std_hat = jnp.sqrt(1 / prec_hat)
mu_hat = (state["lambda"] * state["mu"] + tau * jnp.sum(state["y"])) / prec_hat

fig = go.Figure()
fig.add_trace(
    go.Histogram(
        x=mcmc.store["s"].flatten(),
        histnorm="probability density"
    )
)
knots = np.linspace(mu_hat - 5 * std_hat, mu_hat + 5 * std_hat, 200)
pdf_vals = scipy.stats.norm.pdf(x=knots, loc=mu_hat, scale=std_hat)
fig.add_vline(x=mu_hat)
fig.add_trace(
    go.Scatter(
        x=knots.flatten(),
        y=pdf_vals.flatten(),
        mode="lines",
        name="true density",
        line=dict(color="black")
    )
)
fig.show()