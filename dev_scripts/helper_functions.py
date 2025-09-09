"""Helper functions for the testing of JAX functionality."""

from copy import deepcopy
import numpy as np
import pandas as pd
import datetime

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


def make_sensor_locations(nof_beams: int, nof_points: int, time_points: pd.array) -> SensorGroup:
    """Make default sensor layout"""
    nof_observations = time_points.size
    reference_latitude = 0
    reference_longitude = 0
    reference_altitude = 0

    radius = 30
    angles = np.linspace(0, 90, nof_beams)
    sensor_x = radius * np.cos(angles*np.pi/180)
    sensor_y = radius * np.sin(angles*np.pi/180)
    sensor_z = np.ones_like(sensor_x) * 5.0
    ENU_object = ENU(ref_latitude=reference_latitude, ref_longitude=reference_longitude, ref_altitude=reference_altitude)
    ENU_object.from_array(np.vstack([sensor_x, sensor_y, sensor_z]).T)
    LLA_object = ENU_object.to_lla()
    LLA_array = LLA_object.to_array()
    print(LLA_array)
    nof_beams = LLA_array.shape[0]
    sensor_group = SensorGroup()
    for sensor in range(nof_beams):
        new_sensor = Beam()
        new_sensor.label = f"Beam sensor {sensor}"
        new_sensor.location = LLA(
            latitude=np.array([reference_latitude, LLA_object.latitude[sensor]]),
            longitude=np.array([reference_longitude, LLA_object.longitude[sensor]]),
            altitude=np.array([5.0, LLA_object.altitude[sensor]])
        )
        new_sensor.time = time_points
        new_sensor.concentration = np.zeros(nof_observations)
        sensor_group.add_sensor(new_sensor)

    sensor_x = np.random.uniform(0, radius, nof_points)
    sensor_y = np.random.uniform(0, radius, nof_points)
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
        new_sensor.time = time_points
        new_sensor.concentration = np.zeros(nof_observations)
        sensor_group.add_sensor(new_sensor)

    return sensor_group


def make_meteorology(time_points: pd.array) -> Meteorology:
    """Create synthetic meteorology object."""
    nof_observations = time_points.size
    met_object = Meteorology()
    met_object.time = time_points
    met_object.wind_direction = np.linspace(0.0, 360.0, nof_observations) + np.random.normal(loc=0.0, scale=0.01, size=nof_observations)
    met_object.wind_speed = 4.0 * np.ones_like(met_object.wind_direction) + np.random.normal(loc=0.0, scale=0.01, size=nof_observations)
    met_object.calculate_uv_from_wind_speed_direction()
    met_object.temperature = (273.1 + 15.0) * np.ones_like(met_object.wind_direction)
    met_object.pressure = 101.325 * np.ones_like(met_object.wind_direction)
    met_object.wind_turbulence_horizontal = 5.0 * np.ones_like(met_object.wind_direction)
    met_object.wind_turbulence_vertical = 5.0 * np.ones_like(met_object.wind_direction)
    return met_object


def make_source_map(nof_sources: int) -> SourceMap:
    """Make default source map."""
    reference_latitude = 0
    reference_longitude = 0
    reference_altitude = 0
    source_map = SourceMap()
    site_limits = np.array([[0, 30],
                            [0, 30],
                            [0, 3]])
    location_object = ENU(ref_latitude=reference_latitude, ref_longitude=reference_longitude, ref_altitude=reference_altitude)
    source_map.generate_sources(coordinate_object=location_object, sourcemap_limits=site_limits, sourcemap_type="hypercube", nof_sources=nof_sources)
    return source_map, site_limits


def make_dispersion_model(
        source_map: SourceMap, sensor_group: SensorGroup, met_object: Meteorology, gas_object: CH4
    ) -> GaussianPlume:
    """Create the dispersion model from the previously-specified source map & sensor configuration."""
    dispersion_model = GaussianPlume(source_map=deepcopy(source_map))
    true_emission_rates = np.array([[15.0], [10.0]])
    for current_sensor in sensor_group.values():
        coupling_matrix = dispersion_model.compute_coupling(sensor_object=current_sensor, meteorology_object=met_object,
                                                            gas_object=gas_object, output_stacked=False, run_interpolation=False)
        source_contribution = coupling_matrix @ true_emission_rates
        observation = source_contribution.flatten()
        current_sensor.concentration = observation
    return dispersion_model, sensor_group


def preprocess_data(sensor_group: SensorGroup, met_object: Meteorology, smoothing_period: int = 120) -> Preprocessor:
    analysis_time_range = [datetime.datetime(2024, 1, 1, 8, 0, 0), datetime.datetime(2024, 1, 1, 12, 0, 0)]
    time_bin_edges = pd.array(pd.date_range(analysis_time_range[0], analysis_time_range[1], freq=f'{smoothing_period}s'), dtype='datetime64[ns]')
    prepocessor_object = Preprocessor(time_bin_edges=time_bin_edges, sensor_object=sensor_group, met_object=met_object,
                                    aggregate_function="median")
    min_wind_speed = 0.05
    prepocessor_object.filter_on_met(filter_variable=["wind_speed"], lower_limit=[min_wind_speed], upper_limit=[np.inf])
    return prepocessor_object


def make_default_pyelq_inputs(dispersion_model: GaussianPlume, site_limits: np.ndarray) -> None:
    """Create the default set of pyELQ input settings for testing purposes."""
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

    background_model = SpatioTemporalBackground()
    background_model.n_time = None
    background_model.mean_bg = 2.0
    background_model.spatial_dependence = True
    background_model.initial_precision = 1 / np.power(3e-4, 2)
    background_model.precision_time_0 = 1 / np.power(0.1, 2)
    background_model.spatial_correlation_param = 25.0
    background_model.update_precision = False

    offset_model = PerSensor()
    offset_model.update_precision = False
    offset_model.initial_precision = 1 / (0.001)**2

    error_model = BySensor()
    error_model.initial_precision = 1 / (0.1)**2
    error_model.prior_precision_shape = 1e-2
    error_model.prior_precision_rate = 1e-2

    return source_model, background_model, offset_model, error_model


def generate_pyelq_test_model(
        start_time: str = "2024-01-01 08:00:00",
        end_time: str = "2024-01-01 12:00:00",
        nof_sources: int = 2
    ) -> ELQModel:
    """Create a standard test model with a small number of changeable parameters."""
    time_axis = pd.array(pd.date_range(start=start_time, end=end_time, freq="1s"), dtype='datetime64[ns]')
    # TODO: set this up so we can control the number of data points.
    sensor_group = make_sensor_locations(5, 5, time_axis)
    met_object = make_meteorology(time_axis)
    source_map, site_limits = make_source_map(nof_sources=nof_sources)
    gas_object = CH4()
    dispersion_model, sensor_group = make_dispersion_model(source_map, sensor_group, met_object, gas_object)
    prp_object = preprocess_data(sensor_group=sensor_group, met_object=met_object)
    source_model, background_model, offset_model, error_model = make_default_pyelq_inputs(dispersion_model, site_limits)
    model = ELQModel(
        sensor_object=prp_object.sensor_object,
        meteorology=prp_object.met_object,
        gas_species=gas_object,
        background=background_model,
        source_model=source_model,
        error_model=error_model,
        offset_model=offset_model
    )
    return model