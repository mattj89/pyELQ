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
from pyelq.component.source_model import SourceModelParameter, ScreenedManifoldMALA, SourceReversibleJump, NullSampler, HamiltonianMonteCarlo

from pyelq.coordinate_system import LLA, ENU

""""Generate the data"""

model = generate_pyelq_test_model()
model.n_iter = 1500
model.initialise()

# extract real source locations and emission rates
real_locations = np.concatenate((
    np.atleast_2d(model.components["source"].dispersion_model.source_map.location.east),
    np.atleast_2d(model.components["source"].dispersion_model.source_map.location.north),
    np.atleast_2d(model.components["source"].dispersion_model.source_map.location.up)
), axis=0)
real_emission_rates = np.array([[15.0], [10.0]])
num_real_sources = real_locations.shape[1]
# TODO (03/07/25): make more robust

"""Run the MCMC for the original case"""

model.to_mcmc()
original_state = deepcopy(model.mcmc.state)
model.mcmc.state["z_src"] = np.random.uniform(
    low=np.array([[0], [0], [0]]), high=np.array([[30], [30], [5]]), size=(3, 2)
)
for k in range(2):
    model.mcmc.state = model.components["source"].update_coupling_column(
        model.mcmc.state, update_column=k
    )
model.run_mcmc()
model.from_mcmc()

"""
Configure the data likelihood.
Test that the compilation of the likelihood etc. is working.
"""

# create test state
state = {}

# control max number of sources
model.components["source"].n_sources_max = 10

# choose the starting sources
if False:
    start_locations = real_locations
else:
    start_locations = np.random.uniform(
        low=np.array([[0], [0], [0]]), high=np.array([[30], [30], [5]]), size=(3, real_locations.shape[1])
    )

# populate sources
state["z_src"] = jnp.array(start_locations)
state["n_src"] = state["z_src"].shape[1]
state["s"] = jnp.array(real_emission_rates)

# convert the generated data to jnp
msr_std = 10.0
state["y"] = jnp.array(original_state["y"] + np.random.normal(size=original_state["y"].shape) * msr_std)

# populate initial coupling matrix
state["A"] = jnp.zeros(shape=(state["y"].shape[0], state["n_src"]))

# create predictor object
# form_dict = {"s" + str(i): "A" + str(i) for i in range(model.components["source"].n_sources_max)}
form_dict = {"s": "A"}
source_parameter = SourceModelParameter(
    form=form_dict,
    sensor_object=model.sensor_object,
    meteorology_object=model.meteorology,
    gas_species=model.gas_species,
    source_map=model.components["source"].dispersion_model.source_map,
    n_sources_max=model.components["source"].n_sources_max,
)

# get the coupling columns corresponding to current locations
test_array, _ = source_parameter.predictor(state)
state = source_parameter.update_prefactors(state)

# other params in state
state["Q"] = (1 / msr_std**2) * jnp.eye(state["y"].size) # measurement error precision matrix
state["rho"] = np.array([2.0])  # Poisson rate for the number of sources

# flag for jit compilation
jit_comp_flag = True

# create the data likelihood
likelihood_y = Normal_jax(
    response="y",
    grad_list=["z_src", "s"],
    mean=source_parameter,
    precision=Identity("Q"),
    scalar_precision=1.0 / msr_std**2,
    jit_compile=jit_comp_flag
)
likelihood_y.param_list = ["s", "z_src"]

# test the data likelihood
log_p, state = likelihood_y.log_p(state)

"""Set up the rest of the MCMC sampler components."""

# priors for the sources
state["mu_s"] = jnp.array([0.0])
state["P_s"] = jnp.array([[1.0 / jnp.power(10.0, 2)]])

# model list stuff
model_list = [likelihood_y]
model_list.append(
    Normal_jax(
        response="s",
        grad_list=["s"],
        mean=Identity("mu_s"),
        precision=Identity("P_s"),
        scalar_precision=1 / 50.0**2,
        domain_response_lower=0.0,
        jit_compile=jit_comp_flag
    )
)
model_list[-1].param_list = ["s"]
model_list.append(
    Uniform_jax(
        response="z_src",
        grad_list=["z_src"],
        domain_response_lower=np.array([[0], [0], [0]]),
        domain_response_upper=np.array([[30], [30], [5]]),
    )
)
model_list[-1].param_list = ["z_src"]
# Poisson prior for the number of sources
model_list.append(Poisson(response="n_src", rate="rho"))
# create the overall model
mdl = Model(model_list)
mdl.response = {"y": "mean"}

sampler_list = []
hmc_precision = 0.01
sampler_list.append(NormalNormal("s", mdl, max_variable_size=model.components["source"].n_sources_max))
sampler_list.append(HamiltonianMonteCarlo(
    "z_src", mdl, max_variable_size=(3, model.components["source"].n_sources_max), step=0.01,
    momentum_precision=hmc_precision, epsilon=2.5e-4, num_leapfrog_steps=10
))
sampler_list.append(SourceReversibleJump(
    "n_src", mdl, step=np.array([1.0], ndmin=2),
    n_max=model.components["source"].n_sources_max,
    associated_params=["z_src"]
))

initial_state = deepcopy(state)
mcmc = MCMC(initial_state, sampler_list, model=mdl, n_burn=1000, n_iter=500)
mcmc.run_mcmc()

# NOTE: scaled identity added to the precision matrix in the mMALA sampler, to make it more stable. Should introduce
# proper priors for the source locations and remove this.

"""
Make some plots of the results (both cases).
"""

# choose a number of burn-in

# plot the fit to the data
fig = go.Figure()
data_shape = (len(model.sensor_object), model.sensor_object["Beam sensor 0"].nof_observations)
y_mean = np.reshape(mcmc.store["y"].mean(axis=1), data_shape).T
y_end = np.reshape(mcmc.store["y"][:, -1], data_shape).T
y_std = np.reshape(mcmc.store["y"].std(axis=1), data_shape).T
y_data = np.reshape(initial_state["y"][:, 0], data_shape).T
k = 0
for sensor_key, sensor in model.sensor_object.items():
    fig.add_trace(
        go.Scatter(
            x=sensor.time,
            y=y_data[:, k],
            mode="markers",
            name="Observed data",
            marker=dict(color=model.sensor_object.color_map[k]),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sensor.time,
            y=y_mean[:, k],
            mode="lines",
            name="Mean fit",
            line=dict(color=model.sensor_object.color_map[k])
        )
    )
    k += 1
fig.show()

# plot the traces of the sources
fig = go.Figure()
fig = model.sensor_object.plot_sensor_location(fig)
for i in range(model.components["source"].n_sources_max):
    location_series = mcmc.store["z_src"][:, i, :]
    enu_object = ENU(east=location_series[0, :], north=location_series[1, :], up=location_series[2, :],
                     ref_latitude=0.0, ref_longitude=0.0, ref_altitude=0.0)
    lla_object = enu_object.to_lla()
    emission_series = mcmc.store["s"][i, :]
    fig.add_trace(
            go.Scattermap(
                mode="markers",
                lat=np.array(lla_object.latitude),
                lon=np.array(lla_object.longitude),
                marker=dict(
                    size=10,
                    color=emission_series,
                    coloraxis="coloraxis",
                ),
                showlegend=False
            )
        )
for i in range(real_locations.shape[1]):
    enu_object = ENU(east=real_locations[0, i], north=real_locations[1, i], up=0.0,
                     ref_latitude=0.0, ref_longitude=0.0, ref_altitude=0.0)
    lla_object = enu_object.to_lla()
    fig.add_trace(
        go.Scattermap(
            mode="markers",
            lat=np.array(lla_object.latitude),
            lon=np.array(lla_object.longitude),
            marker=dict(
                size=10,
                color="black"
            ),
            name=f"Real source {i+1}"
        )
    )
fig.update_layout(coloraxis = {'colorscale':'jet'})
fig.show()

"""
Plots of the original MCMC results.
"""

# plot the fit to the data
fig = go.Figure()
data_shape = (len(model.sensor_object), model.sensor_object["Beam sensor 0"].nof_observations)
y_mean_orig = np.reshape(model.mcmc.store["y"][:, 1000:].mean(axis=1), data_shape).T
y_end_orig = np.reshape(model.mcmc.store["y"][:, -1], data_shape).T
y_std_orig = np.reshape(model.mcmc.store["y"][:, 1000:].std(axis=1), data_shape).T
y_data = np.reshape(initial_state["y"][:, 0], data_shape).T
k = 0
for sensor_key, sensor in model.sensor_object.items():
    fig.add_trace(
        go.Scatter(
            x=sensor.time,
            y=y_data[:, k],
            mode="markers",
            name="Observed data",
            marker=dict(color=model.sensor_object.color_map[k]),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sensor.time,
            y=y_mean_orig[:, k],
            mode="lines",
            name="Mean fit",
            line=dict(color=model.sensor_object.color_map[k])
        )
    )
    k += 1
fig.show()

# plot the traces of the sources
fig = go.Figure()
fig = model.sensor_object.plot_sensor_location(fig)
for i in range(model.components["source"].n_sources_max):
    location_series = np.reshape(model.mcmc.store["z_src"][:, i, 1000:], (3, 500))
    enu_object = ENU(east=location_series[0, :], north=location_series[1, :], up=location_series[2, :],
                     ref_latitude=0.0, ref_longitude=0.0, ref_altitude=0.0)
    lla_object = enu_object.to_lla()
    emission_series = model.mcmc.store["s"][i, 1000:]
    fig.add_trace(
            go.Scattermap(
                mode="markers",
                lat=np.array(lla_object.latitude),
                lon=np.array(lla_object.longitude),
                marker=dict(
                    size=10,
                    color=emission_series,
                    coloraxis="coloraxis",
                ),
                showlegend=False
            )
        )
for i in range(real_locations.shape[1]):
    enu_object = ENU(east=real_locations[0, i], north=real_locations[1, i], up=0.0,
                     ref_latitude=0.0, ref_longitude=0.0, ref_altitude=0.0)
    lla_object = enu_object.to_lla()
    fig.add_trace(
        go.Scattermap(
            mode="markers",
            lat=np.array(lla_object.latitude),
            lon=np.array(lla_object.longitude),
            marker=dict(
                size=10,
                color="black"
            ),
            name=f"Real source {i+1}"
        )
    )
fig.update_layout(coloraxis = {'colorscale':'jet'})
fig.show()

"""
Diagnostic plots
"""

# what happens with the num. sources in the solution in both cases?

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=np.arange(model.mcmc.n_iter),
        y=model.mcmc.store["n_src"][0, :],
        mode="lines",
        name="Original MCMC",
        line=dict(color="blue")
    )
)
fig.add_trace(
    go.Scatter(
        x=np.arange(mcmc.n_iter),
        y=mcmc.store["n_src"][0, :],
        mode="lines",
        name="JAX MCMC",
        line=dict(color="red")
    )
)
fig.show()