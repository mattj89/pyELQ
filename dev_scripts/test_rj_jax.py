"""Testing script for the JAX parameter version of the sampler.
"""

import datetime
from copy import deepcopy

import numpy as np
import pandas as pd
import jax.numpy as jnp

import plotly.graph_objects as go
from pyelq.component.background import SpatioTemporalBackground
from pyelq.component.error_model import BySensor
from pyelq.component.offset import PerSensor
from pyelq.component.source_model import Normal
from pyelq.coordinate_system import ENU, LLA
from pyelq.dispersion_model.gaussian_plume import GaussianPlume
from pyelq.gas_species import CH4
from pyelq.model import ELQModel
from pyelq.meteorology import Meteorology
from pyelq.plotting.plot import Plot
from pyelq.preprocessing import Preprocessor
from pyelq.sensor.beam import Beam
from pyelq.sensor.sensor import Sensor, SensorGroup
from pyelq.source_map import SourceMap

from pyelq.component.source_model import SourceModelParameter

from openmcmc import parameter_jax
from openmcmc.distribution.distribution_jax import Normal_jax, Uniform_jax
from openmcmc.distribution import location_scale
from openmcmc import parameter
from openmcmc.model import Model
from openmcmc.sampler.sampler import NormalNormal
from openmcmc.sampler.metropolis_hastings import ManifoldMALA
from openmcmc.sampler.reversible_jump import ReversibleJump
from openmcmc.mcmc import MCMC

"""
Do the data generation.
"""

time_axis = pd.array(pd.date_range(start="2024-01-01 08:00:00", end="2024-01-01 12:00:00", freq="120s"), dtype='datetime64[ns]')
nof_observations = time_axis.size
reference_latitude = 0
reference_longitude = 0
reference_altitude = 0

radius = 30
angles = np.linspace(0, 90, 5)
sensor_x = radius * np.cos(angles*np.pi/180)
sensor_y = radius * np.sin(angles*np.pi/180)
sensor_z = np.ones_like(sensor_x) * 5.0

ENU_object = ENU(ref_latitude=reference_latitude, ref_longitude=reference_longitude, ref_altitude=reference_altitude)
ENU_object.from_array(np.vstack([sensor_x, sensor_y, sensor_z]).T)
LLA_object = ENU_object.to_lla()
LLA_array = LLA_object.to_array()
print(LLA_array)

nof_sensors = LLA_array.shape[0]
sensor_group = SensorGroup()
for sensor in range(nof_sensors):
    new_sensor = Beam()
    new_sensor.label = f"Beam sensor {sensor}"
    new_sensor.location = LLA(
        latitude=np.array([reference_latitude, LLA_object.latitude[sensor]]),
        longitude=np.array([reference_longitude, LLA_object.longitude[sensor]]),
        altitude=np.array([5.0, LLA_object.altitude[sensor]])
    )
    
    new_sensor.time = time_axis
    new_sensor.concentration = np.zeros(nof_observations)
    sensor_group.add_sensor(new_sensor)

sensor_x = np.array([5, 20])
sensor_y = np.array([22, 5])
sensor_z = np.ones_like(sensor_x) * 1.0
ENU_object = ENU(ref_latitude=reference_latitude, ref_longitude=reference_longitude, ref_altitude=reference_altitude)
ENU_object.from_array(np.vstack([sensor_x, sensor_y, sensor_z]).T)
LLA_object = ENU_object.to_lla()
LLA_array = LLA_object.to_array()

nof_sensors = LLA_array.shape[0]
for sensor in range(nof_sensors):
    new_sensor = Sensor()
    new_sensor.label = f"Point sensor {sensor}"
    new_sensor.location = LLA(
        latitude=np.array([LLA_object.latitude[sensor]]),
        longitude=np.array([LLA_object.longitude[sensor]]),
        altitude=np.array([LLA_object.altitude[sensor]])
    )
    
    new_sensor.time = time_axis
    new_sensor.concentration = np.zeros(nof_observations)
    sensor_group.add_sensor(new_sensor)

fig=go.Figure()
fig = sensor_group.plot_sensor_location(fig=fig)
fig.update_layout(mapbox_style="open-street-map", mapbox_center=dict(lat=reference_latitude, lon=reference_longitude),
                  mapbox_zoom=18, height=800, margin={"r":0,"l":0,"b":0})
fig.show()

met_object = Meteorology()

met_object.time = time_axis
met_object.wind_direction = np.linspace(0.0, 90.0, nof_observations) + np.random.normal(loc=0.0, scale=0.1, size=nof_observations)
met_object.wind_speed = 4.0 * np.ones_like(met_object.wind_direction) + np.random.normal(loc=0.0, scale=0.1, size=nof_observations)

met_object.calculate_uv_from_wind_speed_direction()

met_object.temperature = (273.1 + 15.0) * np.ones_like(met_object.wind_direction)
met_object.pressure = 101.325 * np.ones_like(met_object.wind_direction)

met_object.wind_turbulence_horizontal = 5.0 * np.ones_like(met_object.wind_direction)
met_object.wind_turbulence_vertical = 5.0 * np.ones_like(met_object.wind_direction)

fig = met_object.plot_polar_hist()
fig.update_layout(height=400, margin={"r":0,"l":0})
fig.show()

fig = go.Figure()
fig.add_trace(go.Scatter(x=time_axis, y=met_object.wind_direction, mode='markers', name='Wind direction'))
fig.update_layout(height=400, margin={"r":0,"l":0}, title='Wind Direction [deg]')
fig.show()

source_map = SourceMap()
site_limits = np.array([[0, 30],
                        [0, 30],
                        [0, 3]])
location_object = ENU(ref_latitude=reference_latitude, ref_longitude=reference_longitude, ref_altitude=reference_altitude)

# set the max number of sources that can appear in the RJ.
n_sources_max = 10

source_map.generate_sources(coordinate_object=location_object, sourcemap_limits=site_limits, sourcemap_type="hypercube", nof_sources=n_sources_max)

# source_map.location.up = np.array([2.0, 3.0])
# source_map.location.east = np.array([10.0, 20.0])
# source_map.location.north = np.array([20.0, 15.0])

fig=go.Figure()
fig = sensor_group.plot_sensor_location(fig=fig)
fig.update_layout(mapbox_style="open-street-map", mapbox_center=dict(lat=reference_latitude, lon=reference_longitude),
                  mapbox_zoom=18, height=800, margin={"r":0,"l":0,"b":0})
fig.add_trace(go.Scattermapbox(mode="markers",
                               lon=source_map.location.to_lla().longitude,
                               lat=source_map.location.to_lla().latitude,
                               name="True locations",
                               marker=go.scattermapbox.Marker(color="green", size=10))
              )
fig.show()

gas_object = CH4()
dispersion_model = GaussianPlume(source_map=deepcopy(source_map))
true_emission_rates = 2.0 * np.ones(shape=(source_map.nof_sources, 1))
for current_sensor in sensor_group.values():
    coupling_matrix = dispersion_model.compute_coupling(sensor_object=current_sensor, meteorology_object=met_object,
                                                        gas_object=gas_object, output_stacked=False, run_interpolation=False)
    source_contribution = coupling_matrix @ true_emission_rates
    observation = source_contribution.flatten()
    # observation = source_contribution.flatten() + np.random.normal(loc=0.0, scale=0.01, size=current_sensor.nof_observations)
    current_sensor.concentration = observation

fig=go.Figure()
fig = sensor_group.plot_timeseries(fig=fig)
fig.update_layout(height=800, margin={"r":0,"t":10,"l":0,"b":0})
fig.show()

fig = go.Figure()
fig = met_object.plot_polar_scatter(fig=fig, sensor_object=sensor_group)
fig.update_layout(height=400, margin={"r":0,"l":0})
fig.show()

"""
Process the synthetic data
"""

analysis_time_range = [datetime.datetime(2024, 1, 1, 8, 0, 0), datetime.datetime(2024, 1, 1, 12, 0, 0)]

smoothing_period = 10 * 60

time_bin_edges = pd.array(pd.date_range(analysis_time_range[0], analysis_time_range[1], freq=f'{smoothing_period}s'), dtype='datetime64[ns]')

prepocessor_object = Preprocessor(time_bin_edges=time_bin_edges, sensor_object=sensor_group, met_object=met_object,
                                  aggregate_function="median")

min_wind_speed = 0.05
prepocessor_object.filter_on_met(filter_variable=["wind_speed"], lower_limit=[min_wind_speed], upper_limit=[np.infty])

"""
Create the "classic" version of the model.
"""

source_model = Normal()
source_model.emission_rate_mean = np.array([0], ndmin=1)
source_model.initial_precision = np.array([1 / (2.5 ** 2)], ndmin=1)
source_model.reversible_jump = True
source_model.rate_num_sources = 1.0
source_model.dispersion_model = dispersion_model
source_model.update_precision = False
source_model.site_limits = site_limits
source_model.coverage_detection = 0.1  # ppm
source_model.coverage_test_source = 3.0  # kg/hr

background = SpatioTemporalBackground()
background.n_time = None
background.mean_bg = 2.0
background.spatial_dependence = True
background.initial_precision = 1 / np.power(3e-4, 2)
background.precision_time_0 = 1 / np.power(0.1, 2)
background.spatial_correlation_param = 25.0
background.update_precision = False

offset_model = PerSensor()
offset_model.update_precision = False
offset_model.initial_precision = 1 / (0.001)**2

error_model = BySensor()
error_model.initial_precision = 1 / (0.1)**2
error_model.prior_precision_shape = 1e-2
error_model.prior_precision_rate = 1e-2


elq_model = ELQModel(sensor_object=prepocessor_object.sensor_object, meteorology=prepocessor_object.met_object,
                          gas_species=gas_object, background=background, source_model=source_model,
                          error_model=error_model, offset_model=offset_model)
elq_model.n_iter = 1000

elq_model.initialise()
elq_model.to_mcmc()

"""
Create the MCMC model, including the JAX distribution.
"""

# set the max number of sources that can appear in the RJ.
n_sources_max = 10

# set up the response distribution
data_mean = SourceModelParameter(
    form={"s": "A"},
    sensor_object=elq_model.sensor_object,
    meteorology_object=elq_model.meteorology,
    gas_species=elq_model.gas_species,
    source_map=source_map,
)
likelihood_y = Normal_jax(
    response="y", grad_list=["z"], mean=data_mean, precision=parameter_jax.Identity_jax("Q")
)
likelihood_y.param_list = ["s", "z"]

# set up the prior distrbution for the emission rates
prior_s = Normal_jax(
    response="s", grad_list=["s"], 
    mean=parameter_jax.Identity_jax(form="mu_s"), 
    precision=parameter_jax.ScaledMatrix_jax(scalar="lam_s", matrix="P_s")
)
prior_s.param_list = ["s"]

# set up the prior for the location
# prior_z = Uniform_jax(
#     response="z",
#     grad_list=["z"],
#     domain_response_lower=site_limits[:, [0]],
#     domain_response_upper=site_limits[:, [1]],
# )
# prior_z.param_list = ["z"]

# try alternative prior for the location: Normal
prior_z = Normal_jax(
    response="z",
    grad_list=["z"],
    mean=parameter_jax.Identity_jax(form="mu_z"),
    precision=parameter_jax.ScaledMatrix_jax(scalar="lam_z", matrix="P_z")
)
prior_z.param_list = ["z"]

# set up the initial state
initial_state = {"y": jnp.atleast_2d(jnp.array(elq_model.sensor_object.concentration)).T}
# add in the source locations and emission rates
initial_state["z"] = jnp.array(source_map.location.to_array().T)
initial_state["s"] = jnp.array(true_emission_rates)

# set up the remainder of the variables in the state
initial_state["mu_s"] = 0.0 * jnp.ones(shape=(n_sources_max, 1))
initial_state["lam_s"] = jnp.float64(1.0 / jnp.power(100.0, 2))
initial_state["P_s"] = jnp.eye(n_sources_max)
initial_state["Q"] = (1.0 / jnp.power(0.1, 2)) * jnp.eye(initial_state["y"].size)

# prior stuff for z
initial_state["mu_z"] = jnp.zeros(shape=(3, 1))
initial_state["lam_z"] = jnp.float64(1.0 / jnp.power(1000.0, 2))
initial_state["P_z"] = jnp.eye(3)

# set up the "masking" variable for the RJ sampler.
initial_state["n_src"] = 2
initial_state["mask"] = np.zeros(shape=(n_sources_max, 1))
initial_state["mask"][:initial_state["n_src"]] = 1.0

# add in the initial coupling matrix
initial_state = data_mean.update_prefactors(initial_state)

# create the initial y_hat
y_hat, _ = data_mean.predictor(initial_state)
y_hat = np.asarray(y_hat)

# plot alongside original data
fig = go.Figure()
fig = elq_model.sensor_object.plot_timeseries(fig=fig)
plot_index = elq_model.sensor_object.sensor_index
for k, sensor in enumerate(elq_model.sensor_object.values()):
    idx_plot = plot_index == k
    fig.add_trace(
        go.Scatter(x=sensor.time, y=y_hat[idx_plot, 0], mode='lines',
                   name=sensor.label, line=dict(color=elq_model.sensor_object.color_map[k]))
    )
fig.update_layout(height=800, margin={"r":0,"t":10,"l":0,"b":0})
fig.show()
# NOTE (14/06/24): difference here is due to the fact that the JAX version uses the default gradient
# for now.

"""
Set up the MCMC sampler using the above specification.
"""

# put the model together.
likelihood_y.precompute_log_det_precision(initial_state)
mdl = Model([likelihood_y, prior_s, prior_z])

# stuff to make the reversible jump work
matching_params = {"variable": "s", "matrix": "A", "scale": 1.0, "limits": [0.0, 1e6]}

# set up the samplers
sampler = [ManifoldMALA("z", mdl),
           NormalNormal("s", mdl)]
sampler[0].max_variable_size = (3, n_sources_max)
sampler[0].step = np.array([[0.25]])

# make proto birth and death functions for the RJ case to see what happens.
def birth_function_jax(current_state: dict, prop_state: dict):
    """Birth function for the JAX case."""
    prop_state["mask"] = prop_state["mask"].at[prop_state["n_src"] - 1].set(1.0)
    prop_state = data_mean.update_prefactors(prop_state)
    prop_state["s"] = jnp.concatenate((prop_state["s"], jnp.zeros(shape=(1, 1))), axis=0)
    prop_state["P_s"] = jnp.eye(prop_state["s"].shape[0])
    prop_state["mu_s"] = jnp.zeros(shape=prop_state["s"].shape)
    return prop_state, 0.0, 0.0

def death_function_jax(current_state: dict, prop_state: dict, deletion_index: int):
    """Death function for the JAX case.
    
    NOTE (25/04/25): for the moment, this just always deletes the last source. To make it work as a general RJ, we
    need to make the source deletion random.
    """
    prop_state["mask"] = prop_state["mask"].at[prop_state["n_src"] - 1].set(0.0)
    prop_state = data_mean.update_prefactors(prop_state)
    prop_state["s"] = jnp.delete(
        prop_state["s"], deletion_index, axis=0, assume_unique_indices=True
    )
    prop_state["P_s"] = jnp.eye(prop_state["s"].shape[0])
    prop_state["mu_s"] = jnp.zeros(shape=prop_state["s"].shape)
    return prop_state, 0.0, 0.0

# add in the reversible jump sampler for n_src (test to see if/why it breaks)
sampler.append(
    ReversibleJump(
        "n_src",
        mdl,
        step=np.array([1.0], ndmin=2),
        associated_params="z",
        n_max=n_sources_max,
        state_birth_function=birth_function_jax,
        state_death_function=death_function_jax,
        matching_params=matching_params,
    )
)

# perturb the starting point
perturbed_state = deepcopy(initial_state)
# perturbed_state["z"] = perturbed_state["z"] + jnp.array([[1, 1, 0],
#                                                          [1, 1, 0]]).T
# NOTE (06/08/24): If I start this np.sqrt(2) metres away from the source, then it works fine (high acceptance rate).
# If I start 2 metres away without changing the step size, then it gets 0% acceptance.
# The Hessian is position-dependent, so 

# set up the MCMC object
mcmc = MCMC(perturbed_state, sampler, model=mdl, n_burn=2000, n_iter=5000)
mcmc.run_mcmc()

# plot the results
# fig = go.Figure()
# fig = sensor_group.plot_sensor_location(fig=fig)
# fig.update_layout(mapbox_style="open-street-map", mapbox_center=dict(lat=reference_latitude, lon=reference_longitude),
#                   mapbox_zoom=18, height=800, margin={"r":0,"l":0,"b":0})
# fig.add_trace(go.Scattermapbox(mode="markers",
#                                lon=source_map.location.to_lla().longitude,
#                                lat=source_map.location.to_lla().latitude,
#                                name="True locations",
#                                marker=go.scattermapbox.Marker(color="green", size=10))
#               )
# fig.show()

# plot the results in ENU space
fig = go.Figure()
location_array = source_map.location.to_enu().to_array()
for k in range(mcmc.store["z"].shape[1]):
    fig.add_trace(
        go.Scatter(x=mcmc.store["z"][0, k, :].flatten(), y=mcmc.store["z"][1, k, :].flatten(),
                    mode='lines', name="Sampled locations", line=dict(color='blue'))
    )
fig.add_trace(
    go.Scatter(x=location_array[:, 0], y=location_array[:, 1],
               mode='markers', name='True locations', marker=dict(color='green', size=10))
)
fig.show()