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
for i in range(model.components["source"].n_sources_max):
    if i < num_real_sources:
        state["z" + str(i)] = jnp.array(
            np.array([[start_locations[0, i]], [start_locations[1, i]], [start_locations[2, i]]])
        )
    else:
        state["z" + str(i)] = jnp.array(
            np.random.uniform(low=np.array([[0], [0], [0]]), high=np.array([[30], [30], [5]]), size=(3, 1))
        )

# populate emission rates
for i in range(model.components["source"].n_sources_max):
    if i < num_real_sources:
        state["s" + str(i)] = jnp.array([[real_emission_rates[i, 0]]])
    else:
        state["s" + str(i)] = jnp.zeros(shape=(1, 1))

# populate source on/off indicator
state["q"] = jnp.zeros(shape=(model.components["source"].n_sources_max, 1))
for i in range(model.components["source"].n_sources_max):
    if i < 5:
        state["q"] = state["q"].at[i].set(1)
    else:
        state["q"] = state["q"].at[i].set(0)
state["n_src"] = int(jnp.sum(state["q"]))

# convert the generated data to jnp
msr_std = 5.0
state["y"] = jnp.array(original_state["y"] + np.random.normal(size=original_state["y"].shape) * msr_std)

# populate coupling matrix
n_data = state["y"].shape[0]
for i in range(model.components["source"].n_sources_max):
    state["A" + str(i)] = jnp.zeros(shape=(n_data, 1))

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

# get the coupling columns corresponding to current locations
test_array, _ = source_parameter.predictor(state)
for i in range(model.components["source"].n_sources_max):
    state = source_parameter.update_prefactors(state, update_index=i)

# other params in state
state["Q"] = (1 / msr_std**2) * jnp.eye(state["y"].size) # measurement error precision matrix
state["rho"] = np.array([1.0])  # Poisson rate for the number of sources

# flag for jit compilation
jit_comp_flag = True

# create the data likelihood
likelihood_y = Normal_jax(
    response="y",
    grad_list=["z" + str(i) for i in range(model.components["source"].n_sources_max)] + \
        ["s" + str(i) for i in range(model.components["source"].n_sources_max)],
    mean=source_parameter,
    precision=Identity("Q"),
    scalar_precision=1.0 / msr_std**2,
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
            scalar_precision=1 / 50.0**2,
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
hmc_precision = jnp.eye(3) * (0.01)
for i in range(model.components["source"].n_sources_max):
    # sampler_list.append(ScreenedManifoldMALA("z" + str(i), mdl, step=0.5, parameter_index=i))
    sampler_list.append(NormalNormal("s" + str(i), mdl))
    sampler_list.append(HamiltonianMonteCarlo("z" + str(i), mdl, step=0.01, momentum_precision=hmc_precision, epsilon=5e-4, num_leapfrog_steps=20, parameter_index=i))
sampler_list.append(SourceReversibleJump(
    "n_src", mdl, step=np.array([1.0], ndmin=2),
    n_max=model.components["source"].n_sources_max,
    associated_params=["q"]
))
sampler_list.append(NullSampler("q", mdl))

initial_state = deepcopy(state)
mcmc = MCMC(initial_state, sampler_list, model=mdl, n_burn=1000, n_iter=500)
mcmc.run_mcmc()

# NOTE: scaled identity added to the precision matrix in the mMALA sampler, to make it more stable. Should introduce
# proper priors for the source locations and remove this.

"""
Make some plots of the results (both cases).
"""

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
    location_name = "z" + str(i)
    emission_name = "s" + str(i)
    on_index = mcmc.store["q"][i, :] == 1
    location_series = mcmc.store[location_name][:, on_index]
    enu_object = ENU(east=location_series[0, :], north=location_series[1, :], up=location_series[2, :],
                     ref_latitude=0.0, ref_longitude=0.0, ref_altitude=0.0)
    lla_object = enu_object.to_lla()
    emission_series = mcmc.store[emission_name][:, on_index]
    fig.add_trace(
            go.Scattermap(
                mode="markers",
                lat=np.array(lla_object.latitude),
                lon=np.array(lla_object.longitude),
                marker=dict(
                    size=10,
                    color=emission_series[0, :],
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