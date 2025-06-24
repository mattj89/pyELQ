"""
Test the possibility of including individual source in the MCMC separately, to make the JAX compile of the coupling
matrix simpler.

below a change log of things that I had to change in the existing MCMC code in order to make this work.

CHANGELOG:
    2025-05-16: parameter.predictor() signature changes so that state dictionary is always passed as an output argument.
        All of the predictor instances in distribution.py updated to reflect this. In sampler.py, only updated where it
        affected the running of the original MCMC.
    2025-05-16: distribution.log_p() signature changes so that state dictionary is always passed as an output argument.
        Presently I only updated the ones that were touched by the standard pyELQ MCMC. Need to do a comprehensive check
        that all other distribtions are updated. (Also true for location_scale.py etc.)
    2025-05-16: changes in various parts of sampler to reflect the fact that distribution.log_p() is now expected to
        pass an extra output argument.

"""

import datetime
from copy import deepcopy

import numpy as np
import pandas as pd
import jax.numpy as jnp

import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"

from helper_functions import generate_pyelq_test_model

from openmcmc.parameter import Identity
from openmcmc.distribution.distribution import Uniform, Poisson
from openmcmc.distribution.distribution_jax import Normal_jax, Uniform_jax
from openmcmc.model import Model
from openmcmc.sampler.sampler import NormalNormal
from openmcmc.sampler.metropolis_hastings import ManifoldMALA
from openmcmc.mcmc import MCMC
from pyelq.component.source_model import SourceModelParameter, ScreenedManifoldMALA, SourceReversibleJump, NullSampler

""""Generate the data"""

model = generate_pyelq_test_model()
model.n_iter = 20
model.initialise()

"""Run the MCMC for the original case"""

model.to_mcmc()
model.run_mcmc()
model.from_mcmc()

"""
Configure the data likelihood.
Test that the compilcation of the likelihood etc. is working.
"""

# create test state
state = {}

# control max number of sources
model.components["source"].n_sources_max = 10

# populate sources
for i in range(model.components["source"].n_sources_max):
    state["z" + str(i)] = jnp.array(
        np.random.uniform(low=np.array([[0], [0], [0]]), high=np.array([[30], [30], [5]]), size=(3, 1))
    )

# populate emission rates
for i in range(model.components["source"].n_sources_max):
    state["s" + str(i)] = jnp.ones(shape=(1, 1))

# populate source on/off indicator
state["q"] = jnp.zeros(shape=(model.components["source"].n_sources_max, 1))
for i in range(model.components["source"].n_sources_max):
    if i < 5:
        state["q"] = state["q"].at[i].set(1)
    else:
        state["q"] = state["q"].at[i].set(0)
state["n_src"] = int(jnp.sum(state["q"]))

# populate coupling matrix
n_data = 500
for i in range(model.components["source"].n_sources_max):
    state["A" + str(i)] = jnp.ones(shape=(n_data, 1))

# create predictor object
form_dict = {"s" + str(i): "A" + str(i) for i in range(model.components["source"].n_sources_max)}
source_parameter = SourceModelParameter(
    form=form_dict,
    sensor_object=model.sensor_object,
    meteorology_object=model.meteorology,
    gas_species=model.gas_species,
    source_map=model.components["source"].dispersion_model.source_map,
    n_sources_max=model.components["source"].n_sources_max,
)

test_array, _ = source_parameter.predictor(state)
for i in range(model.components["source"].n_sources_max):
    state = source_parameter.update_prefactors(state, update_index=i)
state["y"], _ = source_parameter.predictor(state)

# other params in state
state["Q"] = (10000.0) * jnp.eye(state["y"].size) # measurement error precision matrix
state["rho"] = np.array([2.0])  # Poisson rate for the number of sources

# flag for jit compilation
jit_comp_flag = True

# create the data likelihood
likelihood_y = Normal_jax(
    response="y",
    grad_list=["z" + str(i) for i in range(model.components["source"].n_sources_max)] + \
        ["s" + str(i) for i in range(model.components["source"].n_sources_max)],
    mean=source_parameter,
    precision=Identity("Q"),
    jit_compile=jit_comp_flag
)
likelihood_y.param_list = ["s" + str(i) for i in range(model.components["source"].n_sources_max)] + \
    ["z" + str(i) for i in range(model.components["source"].n_sources_max)]

# test the data likelihood
log_p, state = likelihood_y.log_p(state)

"""Set up the rest of the MCMC sampler components."""

model_list = [likelihood_y]
for i in range(model.components["source"].n_sources_max):
    # prior state stuff
    state["mu_s" + str(i)] = jnp.array([0.0])
    state["P_s" + str(i)] = jnp.array([[1.0 / jnp.power(5.0, 2)]])
    # emission rate prior
    model_list.append(
        Normal_jax(
            response="s" + str(i),
            grad_list=["s" + str(i)],
            mean=Identity("mu_s" + str(i)),
            precision=Identity("P_s" + str(i)),
            domain_response_lower=0.0,
            jit_compile=jit_comp_flag
        )
    )
    model_list[-1].param_list = ["s" + str(i)]
    # location prior
    model_list.append(
        Uniform_jax(
            response="z" + str(i),
            grad_list=["z" + str(i)],
            domain_response_lower=np.array([[0], [0], [0]]),
            domain_response_upper=np.array([[30], [30], [5]]),
        )
    )
    model_list[-1].param_list = ["z" + str(i)]
# Poisson prior for the number of sources
model_list.append(Poisson(response="n_src", rate="rho"))
# create the overall model
mdl = Model(model_list)
mdl.response = {"y": "mean"}

sampler_list = []
for i in range(model.components["source"].n_sources_max):
    sampler_list.append(ScreenedManifoldMALA("z" + str(i), mdl, step=0.2, parameter_index=i))
    sampler_list.append(NormalNormal("s" + str(i), mdl))
sampler_list.append(SourceReversibleJump(
    "n_src", mdl, step=np.array([1.0], ndmin=2),
    n_max=model.components["source"].n_sources_max,
    associated_params=["q"]
))
sampler_list.append(NullSampler("q", mdl))

initial_state = deepcopy(state)
mcmc = MCMC(initial_state, sampler_list, model=mdl, n_burn=10, n_iter=500)
mcmc.run_mcmc()

# NOTE: scaled identity added to the precision matrix in the mMALA sampler, to make it more stable. Should introduce
# proper priors for the source locations and remove this.

# NOTE (13/06/25): leaving it for the afternoon. Notes:
# - currently working through debugger and solving issues as they arise.
# - need to implement Uniform_jax and rvs methods for both this and the Normal_jax dist.
# - Later: check line by line that the right A matrix columns are getting updated when samplers are hit.

"""
Make some plots of the results (both cases).
"""

# plot the fit to the data
fig = go.Figure()
y_mean = mcmc.store["y"].mean(axis=1)
y_std = mcmc.store["y"].std(axis=1)
fig.add_trace(
    go.Scatter(
        x=np.arange(initial_state["y"].shape[0]),
        y=initial_state["y"][:, 0],
        mode="lines",
        name="Observed data",
        line=dict(color='red')
    )
)
fig.add_trace(
    go.Scatter(
        x=np.arange(mcmc.store["y"].shape[0]),
        y=y_mean,
        mode="lines",
        name="Mean fit",
        line=dict(color='blue')
    )
)
fig.show()

# plot the traces of the sources
fig = go.Figure()
fig = model.sensor_object.plot_sensor_location(fig)
for i in range(model.components["source"].n_sources_max):
    location_name = "z" + str(i)
    emission_name = "s" + str(i)
    on_index = mcmc.store["q"][i, :] == 1
    location_series = mcmc.store[location_name][:, on_index]
    emission_series = mcmc.store[emission_name][:, on_index]
    fig.add_trace(
        go.Scatter(
            x=location_series[0, :],
            y=location_series[1, :],
            mode="markers",
            marker=dict(
                size=10,
                color=emission_series[0, :],
                coloraxis="coloraxis",
            ),
            showlegend=False
        )
    )
fig.update_layout(coloraxis = {'colorscale':'jet'})
fig.show()