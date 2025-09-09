"""
Test the possibility of running a simple max likelihood using the newly-available gradients.
"""

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

"""
Generate the data.
"""

model = generate_pyelq_test_model()
model.n_iter = 20
model.initialise()

# extract real source locations and emission rates
real_locations = np.concatenate((
    np.atleast_2d(model.components["source"].dispersion_model.source_map.location.east),
    np.atleast_2d(model.components["source"].dispersion_model.source_map.location.north),
    np.atleast_2d(model.components["source"].dispersion_model.source_map.location.up)
), axis=0)
real_emission_rates = np.array([[15.0], [10.0]])
num_real_sources = real_locations.shape[1]

model.to_mcmc()
original_state = deepcopy(model.mcmc.state)

"""
Configure the data likelihood.
"""

# create test state
state = {}

# control max number of sources
model.components["source"].n_sources_max = 20

# choose the starting sources
if False:
    start_locations = real_locations
else:
    start_locations = np.random.uniform(
        low=np.array([[0], [0], [0]]), high=np.array([[30], [30], [5]]), size=(3, real_locations.shape[1])
    )

# populate sources
for i in range(model.components["source"].n_sources_max):
    state["z" + str(i)] = jnp.array(
        np.random.uniform(low=np.array([[0], [0], [0]]), high=np.array([[30], [30], [5]]), size=(3, 1))
    )

# populate emission rates
for i in range(model.components["source"].n_sources_max):
    state["s" + str(i)] = jnp.zeros(shape=(1, 1))

# populate source on/off indicator
state["q"] = jnp.ones(shape=(model.components["source"].n_sources_max, 1))
# for i in range(model.components["source"].n_sources_max):
#     if i < 5:
#         state["q"] = state["q"].at[i].set(1)
#     else:
#         state["q"] = state["q"].at[i].set(0)
state["n_src"] = int(jnp.sum(state["q"]))

# convert the generated data to jnp
msr_std = 1.0
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

"""
Do a maximum likelihood in a loop.
"""

max_iter = 1000
stop_flag = False
ct = 0
emission_step = 1e-2
location_step = 1e-1

ml_state = deepcopy(state)
log_p = np.zeros(shape=(max_iter,))

while (ct < max_iter) and not stop_flag:
    # # loop over emission rates
    for i in range(model.components["source"].n_sources_max):
        # get conditional predictor
        y_hat_i = mdl["y"].mean.predictor_conditional(ml_state, term_to_exclude="s" + str(i))
        # regress
        s_hat_i = jnp.linalg.solve(
            ml_state["A" + str(i)].T @ ml_state["A" + str(i)] + 1e-4,
            ml_state["A" + str(i)].T @ (ml_state["y"] - y_hat_i)
        )
        ml_state["s" + str(i)] = jnp.maximum(s_hat_i, 0.0)  # ensure non-negative emission rates

    # loop over source emission rates
    # for i in range(model.components["source"].n_sources_max):
    #     # emission rate gradient
    #     grad_s, hess_s = mdl["y"].grad_log_p(ml_state, param="s" + str(i), hessian_required=True)
    #     hess_s += 1e-12 * jnp.eye(hess_s.shape[0])
    #     # make a Newton-Raphson step
    #     ml_state["s" + str(i)] += jnp.maximum(emission_step * jnp.linalg.solve(hess_s, grad_s), 0)

    # loop over source locations
    for i in range(model.components["source"].n_sources_max):
        # source location gradient
        grad_z, hess_z = mdl["y"].grad_log_p(ml_state, param="z" + str(i), hessian_required=True)
        hess_z += 1e-8 * jnp.eye(hess_z.shape[0])
        # make a Newton-Raphson step
        ml_state["z" + str(i)] += location_step * jnp.linalg.solve(hess_z, grad_z)
        # updated A matrix for new location
        _, ml_state = mdl["y"].log_p(ml_state, update_index=i)

    # evaluate likelihood
    log_p[ct], _ = mdl["y"].log_p(ml_state, update_index=i)
    if ct > 0:
        if np.abs(log_p[ct] - log_p[ct - 1]) < 1e-2:
            stop_flag = True
    # print(f"Iteration {ct}: log_p = {log_p[ct]:.3f}")
    if ct > 0:
        print(f"Iteration {ct}: Change in log_p is {log_p[ct] - log_p[ct - 1]:.3f}")
    ct += 1

# calculate the data fit at the final point
data_shape = (len(model.sensor_object), model.sensor_object["Beam sensor 0"].nof_observations)
y_hat_final, _ = mdl["y"].mean.predictor(ml_state)
y_hat_final = np.reshape(y_hat_final, data_shape).T

fig = go.Figure()
y_data = np.reshape(ml_state["y"][:, 0], data_shape).T
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
            y=y_hat_final[:, k],
            mode="lines",
            name="Mean fit",
            line=dict(color=model.sensor_object.color_map[k])
        )
    )
    k += 1
fig.show()

fig = go.Figure()
fig = model.sensor_object.plot_sensor_location(fig)
for i in range(model.components["source"].n_sources_max):
    location_name = "z" + str(i)
    emission_name = "s" + str(i)
    enu_object = ENU(east=ml_state[location_name][0, 0], north=ml_state[location_name][1, 0], up=ml_state[location_name][2, 0],
                     ref_latitude=0.0, ref_longitude=0.0, ref_altitude=0.0)
    lla_object = enu_object.to_lla()
    fig.add_trace(
            go.Scattermap(
                mode="markers",
                lat=np.array(lla_object.latitude),
                lon=np.array(lla_object.longitude),
                marker=dict(
                    size=10,
                    color=ml_state[emission_name],
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
                color="blue"
            ),
            name=f"Real source {i+1}"
        )
    )
fig.update_layout(coloraxis={'colorscale': 'jet'})
fig.show()