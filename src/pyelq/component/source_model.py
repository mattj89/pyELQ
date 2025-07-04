# SPDX-FileCopyrightText: 2024 Shell Global Solutions International B.V. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0

# -*- coding: utf-8 -*-
"""Component Class and subclasses for source model.

A SourceModel instance inherits from 3 super-classes:
    - Component: this is the general superclass for ELQModel components, which prototypes generic methods.
    - A type of SourceGrouping: this class type implements an allocation of sources to different categories (e.g. slab
        or spike), and sets up a sampler for estimating the classification of each source within the source map.
        Inheriting from the NullGrouping class ensures that the allocation of all sources is fixed during the inversion,
        and is not updated.
    - A type of SourceDistribution: this class type implements a particular type of response distribution (mostly
        Normal, but also allows for cases where we have e.g. exp(log_s) or a non-Gaussian prior).

"""

from abc import abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Tuple, Union

import numpy as np
import jax.numpy as jnp

from openmcmc import parameter
from openmcmc import gmrf
from openmcmc.distribution.distribution import Categorical, Gamma, Poisson, Uniform
from openmcmc.distribution.location_scale import Normal as mcmcNormal
from openmcmc.model import Model
from openmcmc.sampler.metropolis_hastings import RandomWalkLoop, ManifoldMALA
from openmcmc.sampler.reversible_jump import ReversibleJump
from openmcmc.sampler.sampler import MixtureAllocation, NormalGamma, NormalNormal, MCMCSampler
from openmcmc.parameter_jax import LinearCombination_jax

from pyelq.component.component import Component
from pyelq.coordinate_system import ENU
from pyelq.dispersion_model.gaussian_plume import GaussianPlume, compute_coupling_array_jax
from pyelq.gas_species import GasSpecies
from pyelq.meteorology import Meteorology
from pyelq.sensor.sensor import SensorGroup, Sensor
from pyelq.sensor.beam import Beam
from pyelq.source_map import SourceMap

if TYPE_CHECKING:
    from pyelq.plotting.plot import Plot


@dataclass
class SourceGrouping:
    """Superclass for source grouping approach.

    Source grouping method determines the group allocation of each source in the model, e.g: slab and spike
    distribution makes an on/off allocation for each source.

    Attributes:
        nof_sources (int): number of sources in the model.
        emission_rate_mean (Union[float, np.ndarray]): prior mean parameter for the emission rate distribution.
        _source_key (str): label for the source parameter to be used in the distributions, samplers, MCMC state etc.

    """

    nof_sources: int = field(init=False)
    emission_rate_mean: Union[float, np.ndarray] = field(init=False)
    _source_key: str = field(init=False, default="s")

    @abstractmethod
    def make_allocation_model(self, model: list) -> list:
        """Initialise the source allocation part of the model, and the parameters of the source response distribution.

        Args:
            model (list): overall model, consisting of list of distributions.

        Returns:
            list: overall model list, updated with allocation distribution.

        """

    @abstractmethod
    def make_allocation_sampler(self, model: Model, sampler_list: list) -> list:
        """Initialise the allocation part of the sampler.

        Args:
            model (Model): overall model, consisting of list of distributions.
            sampler_list (list): list of samplers for individual parameters.

        Returns:
            list: sampler_list updated with sampler for the source allocation.

        """

    @abstractmethod
    def make_allocation_state(self, state: dict) -> dict:
        """Initialise the allocation part of the state.

        Args:
            state (dict): dictionary containing current state information.

        Returns:
            dict: state updated with parameters related to the source grouping.

        """

    @abstractmethod
    def from_mcmc_group(self, store: dict):
        """Extract posterior allocation samples from the MCMC sampler, attach them to the class.

        Args:
            store (dict): dictionary containing samples from the MCMC.

        """


@dataclass
class NullGrouping(SourceGrouping):
    """Null grouping: the grouping of the sources will not change during the course of the inversion.

    Note that this is intended to support two distinct cases:
        1) The case where the source map is fixed, and a given prior mean and prior precision value are assigned to
            each source (can be a common value for all sources, or can be a distinct allocation to each element of the
            source map).
        2) The case where the dimensionality of the source map is changing during the inversion, and a common prior
            mean and precision term are used for all sources.

    """

    def make_allocation_model(self, model: list) -> list:
        """Initialise the source allocation part of the model.

        In the NullGrouping case, the source allocation is fixed throughout, so this function does nothing (simply
        returns the existing model un-modified).

        Args:
            model (list): model as constructed so far, consisting of list of distributions.

        Returns:
            list: overall model list, updated with allocation distribution.

        """
        return model

    def make_allocation_sampler(self, model: Model, sampler_list: list) -> list:
        """Initialise the allocation part of the sampler.

        In the NullGrouping case, the source allocation is fixed throughout, so this function does nothing (simply
        returns the existing sampler list un-modified).

        Args:
            model (Model): overall model set for the problem.
            sampler_list (list): list of samplers for individual parameters.

        Returns:
            list: sampler_list updated with sampler for the source allocation.

        """
        return sampler_list

    def make_allocation_state(self, state: dict) -> dict:
        """Initialise the allocation part of the state.

        The prior mean parameter and the fixed source allocation are added to the state.

        Args:
            state (dict): dictionary containing current state information.

        Returns:
            dict: state updated with parameters related to the source grouping.

        """
        state["mu_s"] = np.array(self.emission_rate_mean, ndmin=1)
        state["alloc_s"] = np.zeros((self.nof_sources, 1), dtype="int")
        return state

    def from_mcmc_group(self, store: dict):
        """Extract posterior allocation samples from the MCMC sampler, attach them to the class.

        We have not implemented anything here as there is nothing to fetch from the MCMC solution here for the
        NullGrouping Class.

        Args:
            store (dict): dictionary containing samples from the MCMC.

        """


@dataclass
class SlabAndSpike(SourceGrouping):
    """Slab and spike source model, special case for the source grouping.

    Slab and spike: the prior for the emission rates is a two-component mixture, and the allocation is to be
    estimated as part of the inversion.

    Attributes:
        slab_probability (float): prior probability of allocation to the slab component. Defaults to 0.05.
        allocation (np.ndarray): set of allocation samples, with shape=(n_sources, n_iterations). Attached to
            the class by self.from_mcmc_group().

    """

    slab_probability: float = 0.05
    allocation: np.ndarray = field(init=False)

    def make_allocation_model(self, model: list) -> list:
        """Initialise the source allocation part of the model.

        Args:
            model (list): model as constructed so far, consisting of list of distributions.

        Returns:
            list: overall model list, updated with allocation distribution.

        """
        model.append(Categorical("alloc_s", prob="s_prob"))
        return model

    def make_allocation_sampler(self, model: Model, sampler_list: list) -> list:
        """Initialise the allocation part of the sampler.

        Args:
            model (Model): overall model set for the problem.
            sampler_list (list): list of samplers for individual parameters.

        Returns:
            list: sampler_list updated with sampler for the source allocation.

        """
        sampler_list.append(MixtureAllocation(param="alloc_s", model=model, response_param=self._source_key))
        return sampler_list

    def make_allocation_state(self, state: dict) -> dict:
        """Initialise the allocation part of the state.

        Args:
            state (dict): dictionary containing current state information.

        Returns:
            dict: state updated with parameters related to the source grouping.

        """
        state["mu_s"] = np.array(self.emission_rate_mean, ndmin=1)
        state["s_prob"] = np.tile(np.array([self.slab_probability, 1 - self.slab_probability]), (self.nof_sources, 1))
        state["alloc_s"] = np.ones((self.nof_sources, 1), dtype="int")
        return state

    def from_mcmc_group(self, store: dict):
        """Extract posterior allocation samples from the MCMC sampler, attach them to the class.

        Args:
            store (dict): dictionary containing samples from the MCMC.

        """
        self.allocation = store["alloc_s"]


@dataclass
class SourceDistribution:
    """Superclass for source emission rate distribution.

    Source distribution determines the type of prior to be used for the source emission rates, and the transformation
    linking the source parameters and the data.

    Elements related to transformation of source parameters are also specified at the model level.

    Attributes:
        nof_sources (int): number of sources in the model.
        emission_rate (np.ndarray): set of emission rate samples, with shape=(n_sources, n_iterations). Attached to
            the class by self.from_mcmc_dist().

    """

    nof_sources: int = field(init=False)
    emission_rate: np.ndarray = field(init=False)

    @abstractmethod
    def make_source_model(self, model: list) -> list:
        """Add distributional component to the overall model corresponding to the source emission rate distribution.

        Args:
            model (list): model as constructed so far, consisting of list of distributions.

        Returns:
            list: overall model list, updated with distributions related to source prior.

        """

    @abstractmethod
    def make_source_sampler(self, model: Model, sampler_list: list) -> list:
        """Initialise the source prior distribution part of the sampler.

        Args:
            model (Model): overall model set for the problem.
            sampler_list (list): list of samplers for individual parameters.

        Returns:
            list: sampler_list updated with sampler for the emission rate parameters.

        """

    @abstractmethod
    def make_source_state(self, state: dict) -> dict:
        """Initialise the emission rate parts of the state.

        Args:
            state (dict): dictionary containing current state information.

        Returns:
            dict: state updated with parameters related to the source emission rates.

        """

    @abstractmethod
    def from_mcmc_dist(self, store: dict):
        """Extract posterior emission rate samples from the MCMC, attach them to the class.

        Args:
            store (dict): dictionary containing samples from the MCMC.

        """


@dataclass
class NormalResponse(SourceDistribution):
    """(Truncated) Gaussian prior for sources.

    No transformation applied to parameters, i.e.:
    - Prior distribution: s ~ N(mu, 1/precision)
    - Likelihood contribution: y = A*s + b + ...

    Attributes:
        truncation (bool): indication of whether the emission rate prior should be truncated at 0. Defaults to True.
        emission_rate_lb (Union[float, np.ndarray]): lower bound for the source emission rates. Defaults to 0.
        emission_rate_mean (Union[float, np.ndarray]): prior mean for the emission rate distribution. Defaults to 0.

    """

    truncation: bool = True
    emission_rate_lb: Union[float, np.ndarray] = 0
    emission_rate_mean: Union[float, np.ndarray] = 0

    def make_source_model(self, model: list) -> list:
        """Add distributional component to the overall model corresponding to the source emission rate distribution.

        Args:
            model (list): model as constructed so far, consisting of list of distributions.

        Returns:
            list: model, updated with distributions related to source prior.

        """
        domain_response_lower = None
        if self.truncation:
            domain_response_lower = self.emission_rate_lb

        model.append(
            mcmcNormal(
                "s",
                mean=parameter.MixtureParameterVector(param="mu_s", allocation="alloc_s"),
                precision=parameter.MixtureParameterMatrix(param="lambda_s", allocation="alloc_s"),
                domain_response_lower=domain_response_lower,
            )
        )
        return model

    def make_source_sampler(self, model: Model, sampler_list: list = None) -> list:
        """Initialise the source prior distribution part of the sampler.

        Args:
            model (Model): overall model set for the problem.
            sampler_list (list): list of samplers for individual parameters.

        Returns:
            list: sampler_list updated with sampler for the emission rate parameters.

        """
        if sampler_list is None:
            sampler_list = []
        sampler_list.append(NormalNormal("s", model))
        return sampler_list

    def make_source_state(self, state: dict) -> dict:
        """Initialise the emission rate part of the state.

        Args:
            state (dict): dictionary containing current state information.

        Returns:
            dict: state updated with initial emission rate vector.

        """
        state["s"] = np.zeros((self.nof_sources, 1))
        return state

    def from_mcmc_dist(self, store: dict):
        """Extract posterior emission rate samples from the MCMC sampler, attach them to the class.

        Args:
            store (dict): dictionary containing samples from the MCMC.

        """
        self.emission_rate = store["s"]


@dataclass
class SourceModel(Component, SourceGrouping, SourceDistribution):
    """Superclass for the specification of the source model in an inversion run.

    Various different types of model. A SourceModel is an optional component of a model, and thus inherits
    from Component.

    A subclass instance of SourceModel must inherit from:
        - an INSTANCE of SourceDistribution, which specifies a prior emission rate distribution for all sources in the
            source map.
        - an INSTANCE of SourceGrouping, which specifies a type of mixture prior specification for the sources (for
            which the allocation is to be estimated as part of the inversion).

    If the flag reversible_jump == True, then the number of sources and their locations are also estimated as part of
    the inversion, in addition to the emission rates. If this flag is set to true, the sensor_object, meteorology and
    gas_species objects are all attached to the class, as they will be required in the repeated computation of updates
    to the coupling matrix during the inversion.

    Attributes:
        dispersion_model (GaussianPlume): dispersion model used to generate the couplings between source locations and
            sensor observations.
        coupling (np.ndarray): coupling matrix generated using dispersion_model.

        sensor_object (SensorGroup): stores sensor information for reversible jump coupling updates.
        meteorology (MeteorologyGroup): stores meteorology information for reversible jump coupling updates.
        gas_species (GasSpecies): stores gas species information for reversible jump coupling updates.

        reversible_jump (bool): logical indicating whether the reversible jump algorithm for estimation of the number
            of sources and their locations should be run. Defaults to False.
        random_walk_step_size (np.ndarray): (3 x 1) array specifying the standard deviations of the distributions
            from which the random walk sampler draws new source locations. Defaults to np.array([1.0, 1.0, 0.1]).
        site_limits (np.ndarray): (3 x 2) array specifying the lower (column 0) and upper (column 1) limits of the
            analysis site. Only relevant for cases where reversible_jump == True (where sources are free to move in
            the solution).
        rate_num_sources (int): specification for the parameter for the Poisson prior distribution for the total number
            of sources. Only relevant for cases where reversible_jump == True (where the number of sources in the
            solution can change).
        n_sources_max (int): maximum number of sources that can feature in the solution. Only relevant for cases where
            reversible_jump == True (where the number of sources in the solution can change).
        emission_proposal_std (float): standard deviation of the truncated Gaussian distribution used to propose the
            new source emission rate in case of a birth move.

        update_precision (bool): logical indicating whether the prior precision parameter for emission rates should be
            updated as part of the inversion. Defaults to false.
        prior_precision_shape (Union[float, np.ndarray]): shape parameters for the prior Gamma distribution for the
            source precision parameter.
        prior_precision_rate (Union[float, np.ndarray]): rate parameters for the prior Gamma distribution for the
            source precision parameter.
        initial_precision (Union[float, np.ndarray]): initial value for the source emission rate precision parameter.
        precision_scalar (np.ndarray): precision values generated by MCMC inversion.

        coverage_detection (float): sensor detection threshold (in ppm) to be used for coverage calculations.
        coverage_test_source (float): test source (in kg/hr) which we wish to be able to see in coverage calculation.

        threshold_function (Callable): Callable function which returns a single value that defines the threshold
            for the coupling in a lambda function form. Examples: lambda x: np.quantile(x, 0.95, axis=0),
            lambda x: np.max(x, axis=0), lambda x: np.mean(x, axis=0). Defaults to np.quantile.

    """

    dispersion_model: GaussianPlume = field(init=False, default=None)
    coupling: np.ndarray = field(init=False)

    sensor_object: SensorGroup = field(init=False, default=None)
    meteorology: Meteorology = field(init=False, default=None)
    gas_species: GasSpecies = field(init=False, default=None)

    reversible_jump: bool = False
    random_walk_step_size: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 0.1], ndmin=2).T)
    site_limits: np.ndarray = None
    rate_num_sources: int = 5
    n_sources_max: int = 20
    emission_proposal_std: float = 0.5

    update_precision: bool = False
    prior_precision_shape: Union[float, np.ndarray] = 1e-3
    prior_precision_rate: Union[float, np.ndarray] = 1e-3
    initial_precision: Union[float, np.ndarray] = 1.0
    precision_scalar: np.ndarray = field(init=False)

    coverage_detection: float = 0.1
    coverage_test_source: float = 6.0

    threshold_function: callable = lambda x: np.quantile(x, 0.95, axis=0)

    @property
    def nof_sources(self):
        """Get number of sources in the source map."""
        return self.dispersion_model.source_map.nof_sources

    @property
    def coverage_threshold(self):
        """Compute coverage threshold from detection threshold and test source strength."""
        return self.coverage_test_source / self.coverage_detection

    def initialise(self, sensor_object: SensorGroup, meteorology: Meteorology, gas_species: GasSpecies):
        """Set up the source model.

        Extract required information from the sensor, meteorology and gas species objects:
            - Attach coupling calculated using self.dispersion_model.
            - (If self.reversible_jump == True) Attach objects to source model which will be used in RJMCMC sampler,
                they will be required when we need to update the couplings when new source locations are proposed when
                we move/birth/death.

        Args:
            sensor_object (SensorGroup): object containing sensor data.
            meteorology (MeteorologyGroup): object containing meteorology data.
            gas_species (GasSpecies): object containing gas species information.

        """
        self.initialise_dispersion_model(sensor_object)
        self.coupling = self.dispersion_model.compute_coupling(
            sensor_object, meteorology, gas_species, output_stacked=True
        )
        self.screen_coverage()
        if self.reversible_jump:
            self.sensor_object = sensor_object
            self.meteorology = meteorology
            self.gas_species = gas_species

    def initialise_dispersion_model(self, sensor_object: SensorGroup):
        """Initialise the dispersion model.

        If a dispersion_model has already been attached to this instance, then this function takes no action.

        If a dispersion_model has not already been attached to the instance, then this function adds a GaussianPlume
        dispersion model, with a default source map that has limits set based on the sensor locations.

        Args:
            sensor_object (SensorGroup): object containing sensor data.

        """
        if self.dispersion_model is None:
            source_map = SourceMap()
            sensor_locations = sensor_object.location.to_enu()
            location_object = ENU(
                ref_latitude=sensor_locations.ref_latitude,
                ref_longitude=sensor_locations.ref_longitude,
                ref_altitude=sensor_locations.ref_altitude,
            )
            source_map.generate_sources(
                coordinate_object=location_object,
                sourcemap_limits=np.array(
                    [
                        [np.min(sensor_locations.east), np.max(sensor_locations.east)],
                        [np.min(sensor_locations.north), np.max(sensor_locations.north)],
                        [np.min(sensor_locations.up), np.max(sensor_locations.up)],
                    ]
                ),
                sourcemap_type="grid",
            )
            self.dispersion_model = GaussianPlume(source_map)

    def screen_coverage(self):
        """Screen the initial source map for coverage."""
        in_coverage_area = self.dispersion_model.compute_coverage(
            self.coupling, coverage_threshold=self.coverage_threshold, threshold_function=self.threshold_function
        )
        self.coupling = self.coupling[:, in_coverage_area]
        all_locations = self.dispersion_model.source_map.location.to_array()
        screened_locations = all_locations[in_coverage_area, :]
        self.dispersion_model.source_map.location.from_array(screened_locations)

    def update_coupling_column(self, state: dict, update_column: int) -> dict:
        """Update the coupling, based on changes to the source locations as part of inversion.

        To be used in two different situations:
            - movement of source locations (e.g. Metropolis Hastings, random walk).
            - adding of new source locations (e.g. reversible jump birth move).
        If [update_column < A.shape[1]]: an existing column of the A matrix is updated.
        If [update_column == A.shape[1]]: a new column is appended to the right-hand side of the A matrix
        (corresponding to a new source).

        A central assumption of this function is that the sensor information and meteorology information
        have already been interpolated onto the same space/time points.

        If an update_column is supplied, the coupling for that source location only is calculated to save on
        computation time. If update_column is None, then we just re-compute the whole coupling matrix.

        Args:
            state (dict): dictionary containing state parameters.
            update_column (int): index of the coupling column to be updated.

        Returns:
            state (dict): state dictionary containing updated coupling information.

        """
        self.dispersion_model.source_map.location.from_array(state["z_src"][:, [update_column]].T)
        new_coupling = self.dispersion_model.compute_coupling(
            self.sensor_object, self.meteorology, self.gas_species, output_stacked=True, run_interpolation=False
        )

        if update_column == state["A"].shape[1]:
            state["A"] = np.concatenate((state["A"], new_coupling), axis=1)
        elif update_column < state["A"].shape[1]:
            state["A"][:, [update_column]] = new_coupling
        else:
            raise ValueError("Invalid column specification for updating.")
        return state

    def birth_function(self, current_state: dict, prop_state: dict) -> Tuple[dict, float, float]:
        """Update MCMC state based on source birth proposal.

        Proposed state updated as follows:
            1- Add column to coupling matrix for new source location.
            2- If required, adjust other components of the state which correspond to the sources.
        The source emission rate vector will be adjusted using the standardised functionality
        in the openMCMC package.

        After the coupling has been updated, a coverage test is applied for the new source
        location. If the max coupling is too small, a large contribution is added to the
        log-proposal density for the new state, to force the sampler to reject it.

        A central assumption of this function is that the sensor information and meteorology information
        have already been interpolated onto the same space/time points.

        This function assumes that the new source location has been added as the final column of
        the source location matrix, and so will correspondingly append the new coupling column to the right
        hand side of the current state coupling, and append an emission rate as the last element of the
        current state emission rate vector.

        Args:
            current_state (dict): dictionary containing parameters of the current state.
            prop_state (dict): dictionary containing the parameters of the proposed state.

        Returns:
            prop_state (dict): proposed state, with coupling matrix and source emission rate vector updated.
            logp_pr_g_cr (float): log-transition density of the proposed state given the current state
                (i.e. log[p(proposed | current)])
            logp_cr_g_pr (float): log-transition density of the current state given the proposed state
                (i.e. log[p(current | proposed)])

        """
        prop_state = self.update_coupling_column(prop_state, int(prop_state["n_src"]) - 1)
        prop_state["alloc_s"] = np.concatenate((prop_state["alloc_s"], np.array([0], ndmin=2)), axis=0)
        in_cov_area = self.dispersion_model.compute_coverage(
            prop_state["A"][:, -1],
            coverage_threshold=self.coverage_threshold,
            threshold_function=self.threshold_function,
        )
        if not in_cov_area:
            logp_pr_g_cr = 1e10
        else:
            logp_pr_g_cr = 0.0
        logp_cr_g_pr = 0.0

        return prop_state, logp_pr_g_cr, logp_cr_g_pr

    @staticmethod
    def death_function(current_state: dict, prop_state: dict, deletion_index: int) -> Tuple[dict, float, float]:
        """Update MCMC state based on source death proposal.

        Proposed state updated as follows:
            1- Remove column from coupling for deleted source.
            2- If required, adjust other components of the state which correspond to the sources.
        The source emission rate vector will be adjusted using the standardised functionality in the general_mcmc repo.

        A central assumption of this function is that the sensor information and meteorology information have already
        been interpolated onto the same space/time points.

        Args:
            current_state (dict): dictionary containing parameters of the current state.
            prop_state (dict): dictionary containing the parameters of the proposed state.
            deletion_index (int): index of the source to be deleted in the overall set of sources.

        Returns:
            prop_state (dict): proposed state, with coupling matrix and source emission rate vector updated.
            logp_pr_g_cr (float): log-transition density of the proposed state given the current state
                (i.e. log[p(proposed | current)])
            logp_cr_g_pr (float): log-transition density of the current state given the proposed state
                (i.e. log[p(current | proposed)])

        """
        prop_state["A"] = np.delete(prop_state["A"], obj=deletion_index, axis=1)
        prop_state["alloc_s"] = np.delete(prop_state["alloc_s"], obj=deletion_index, axis=0)
        logp_pr_g_cr = 0.0
        logp_cr_g_pr = 0.0

        return prop_state, logp_pr_g_cr, logp_cr_g_pr

    def move_function(self, current_state: dict, update_column: int) -> dict:
        """Re-compute the coupling after a source location move.

        Function first updates the coupling column, and then checks whether the location passes a coverage test. If the
        location does not have good enough coverage, the state reverts to the coupling from the current state.

        Args:
            current_state (dict): dictionary containing parameters of the current state.
            update_column (int): index of the coupling column to be updated.

        Returns:
            dict: proposed state, with updated coupling matrix.

        """
        prop_state = deepcopy(current_state)
        prop_state = self.update_coupling_column(prop_state, update_column)
        in_cov_area = self.dispersion_model.compute_coverage(
            prop_state["A"][:, update_column],
            coverage_threshold=self.coverage_threshold,
            threshold_function=self.threshold_function,
        )
        if not in_cov_area:
            prop_state = deepcopy(current_state)
        return prop_state

    def make_model(self, model: list) -> list:
        """Take model list and append new elements from current model component.

        Args:
            model (list): Current list of model elements.

        Returns:
            list: model list updated with source-related distributions.

        """
        model = self.make_allocation_model(model)
        model = self.make_source_model(model)
        if self.update_precision:
            model.append(Gamma("lambda_s", shape="a_lam_s", rate="b_lam_s"))
        if self.reversible_jump:
            model.append(
                Uniform(
                    response="z_src",
                    domain_response_lower=self.site_limits[:, [0]],
                    domain_response_upper=self.site_limits[:, [1]],
                )
            )
            model.append(Poisson(response="n_src", rate="rho"))
        return model

    def make_sampler(self, model: Model, sampler_list: list) -> list:
        """Take sampler list and append new elements from current model component.

        Args:
            model (Model): Full model list of distributions.
            sampler_list (list): Current list of samplers.

        Returns:
            list: sampler list updated with source-related samplers.

        """
        sampler_list = self.make_source_sampler(model, sampler_list)
        sampler_list = self.make_allocation_sampler(model, sampler_list)
        if self.update_precision:
            sampler_list.append(NormalGamma("lambda_s", model))
        if self.reversible_jump:
            sampler_list = self.make_sampler_rjmcmc(model, sampler_list)
        return sampler_list

    def make_state(self, state: dict) -> dict:
        """Take state dictionary and append initial values from model component.

        Args:
            state (dict): current state vector.

        Returns:
            dict: current state vector with source-related parameters added.

        """
        state = self.make_allocation_state(state)
        state = self.make_source_state(state)
        state["A"] = self.coupling
        state["lambda_s"] = np.array(self.initial_precision, ndmin=1)
        if self.update_precision:
            state["a_lam_s"] = np.ones_like(self.initial_precision) * self.prior_precision_shape
            state["b_lam_s"] = np.ones_like(self.initial_precision) * self.prior_precision_rate
        if self.reversible_jump:
            state["z_src"] = self.dispersion_model.source_map.location.to_array().T
            state["n_src"] = state["z_src"].shape[1]
            state["rho"] = self.rate_num_sources
        return state

    def make_sampler_rjmcmc(self, model: Model, sampler_list: list) -> list:
        """Create the parts of the sampler related to the reversible jump MCMC scheme.

        RJ MCMC scheme:
            - create the RandomWalkLoop sampler object which updates the source locations one-at-a-time.
            - create the ReversibleJump sampler which proposes birth/death moves to add/remove sources from the source
                map.

        Args:
            model (Model): model object containing probability density objects for all uncertain
                parameters.
            sampler_list (list): list of existing samplers.

        Returns:
            sampler_list (list): list of samplers updated with samplers corresponding to RJMCMC routine.

        """
        sampler_list[-1].max_variable_size = self.n_sources_max

        sampler_list.append(
            RandomWalkLoop(
                "z_src",
                model,
                step=self.random_walk_step_size,
                max_variable_size=(3, self.n_sources_max),
                domain_limits=self.site_limits,
                state_update_function=self.move_function,
            )
        )
        matching_params = {"variable": "s", "matrix": "A", "scale": 1.0, "limits": [0.0, 1e6]}
        sampler_list.append(
            ReversibleJump(
                "n_src",
                model,
                step=np.array([1.0], ndmin=2),
                associated_params="z_src",
                n_max=self.n_sources_max,
                state_birth_function=self.birth_function,
                state_death_function=self.death_function,
                matching_params=matching_params,
            )
        )
        return sampler_list

    def from_mcmc(self, store: dict):
        """Extract results of mcmc from mcmc.store and attach to components.

        Args:
            store (dict): mcmc result dictionary.

        """
        self.from_mcmc_group(store)
        self.from_mcmc_dist(store)
        if self.update_precision:
            self.precision_scalar = store["lambda_s"]

    def plot_iterations(self, plot: "Plot", burn_in_value: int, y_axis_type: str = "linear") -> "Plot":
        """Plot the emission rate estimates source model object against MCMC iteration.

        Args:
            burn_in_value (int): Burn in value to show in plot.
            y_axis_type (str, optional): String to indicate whether the y-axis should be linear of log scale.
            plot (Plot): Plot object to which this figure will be added in the figure dictionary.

        Returns:
            plot (Plot): Plot object to which the figures added in the figure dictionary with
                keys 'estimated_values_plot'/'log_estimated_values_plot' and 'number_of_sources_plot'

        """
        plot.plot_emission_rate_estimates(source_model_object=self, burn_in=burn_in_value, y_axis_type=y_axis_type)
        plot.plot_single_trace(object_to_plot=self)
        return plot


@dataclass
class Normal(SourceModel, NullGrouping, NormalResponse):
    """Normal model, with null allocation.

    (Truncated) Gaussian prior for emission rates, no grouping/allocation; no transformation applied to emission rate
    parameters.

    Can be used in the following cases:
        - Fixed set of sources (grid or specific locations), all with the same Gaussian prior distribution.
        - Variable number of sources, with a common prior distribution, estimated using reversible jump MCMC.
        - Fixed set of sources with a bespoke prior per source (using the allocation to map prior parameters onto
            sources).

    """


@dataclass
class NormalSlabAndSpike(SourceModel, SlabAndSpike, NormalResponse):
    """Normal Slab and Spike model.

    (Truncated) Gaussian prior for emission rates, slab and spike prior, with allocation estimation; no transformation
    applied to emission rate parameters.

    Attributes:
        initial_precision (np.ndarray): initial precision parameter for a slab and spike case. shape=(2, 1).
        emission_rate_mean (np.ndarray): emission rate prior mean for a slab and spike case. shape=(2, 1).

    """

    initial_precision: np.ndarray = field(default_factory=lambda: np.array([1 / (10**2), 1 / (0.01**2)], ndmin=2).T)
    emission_rate_mean: np.ndarray = field(default_factory=lambda: np.array([0, 0], ndmin=2).T)


@dataclass
class SourceModelParameter(LinearCombination_jax):
    """Parameter class for the source model, which allows for updating of the coupling matrix as part of the
    log-likelihood calls.

    Initialization of the parameter class extracts the sensor locations and meteorology information from the objects
    passed in, converts them to jax.numpy arrays and stores them locally on the class.

    Attributes:
        sensor_locations (dict):

    """
    sensor_locations_x: dict
    sensor_locations_y: dict
    sensor_locations_z: dict
    form: dict
    wind_speed: jnp.ndarray
    theta: jnp.ndarray
    wind_turbulence_horizontal: jnp.ndarray
    wind_turbulence_vertical: jnp.ndarray
    gas_density: jnp.ndarray
    n_sources_max: int

    def __init__(self, form, sensor_object, meteorology_object, gas_species, source_map, n_sources_max):
        """Function which takes the pyELQ data sensor and meteorology objects, converts the relevant attributes to
        jax.numpy objects, and attaches them to the class ready for repeated use in the sampler.



        """
        self.form = form
        self.extract_sensor_information(sensor_object, source_map)
        self.extract_meteorology_information(meteorology_object)
        self.gas_density = jnp.array(gas_species.gas_density())
        self.n_sources_max = n_sources_max

    def predictor_conditional(self, state, term_to_exclude = None):
        """Overloaded version, to take account of the fact that the terms are being screened in/out by the RJ
        indicator.
        """
        if term_to_exclude is None:
            term_to_exclude = []

        if isinstance(term_to_exclude, str):
            term_to_exclude = [term_to_exclude]

        sum_terms = 0
        ct = 0
        for prm, prefactor in self.form.items():
            if prm not in term_to_exclude:
                sum_terms += state["q"][ct] * (state[prefactor] @ state[prm])
            ct += 1
        # TODO (17/06/25): robustify this counting mechanism.
        return sum_terms

    def extract_sensor_information(self, sensor_object, source_map):
        """Sub-function for extracting and storing sensor location information."""
        source_map_enu = source_map.location
        # TODO (14/06/24): May need to handle the co-ordinate conversion- assuming ENU for now.
        self.sensor_locations_x = {}
        self.sensor_locations_y = {}
        self.sensor_locations_z = {}
        for key, sensor in sensor_object.items():
            if isinstance(sensor, Beam):
                enu_sensor_array = sensor.make_beam_knots(
                    ref_latitude=source_map_enu.ref_latitude,
                    ref_longitude=source_map_enu.ref_longitude,
                    ref_altitude=source_map_enu.ref_altitude,
                )
                enu_sensor_array = np.swapaxes(np.atleast_3d(enu_sensor_array), 0, 2)
            else:
                enu_sensor_array = sensor.location.to_enu(
                    ref_latitude=source_map_enu.ref_latitude,
                    ref_longitude=source_map_enu.ref_longitude,
                    ref_altitude=source_map_enu.ref_altitude,
                ).to_array()
                enu_sensor_array = np.atleast_3d(enu_sensor_array)
            jax_locations = jnp.array(enu_sensor_array)
            self.sensor_locations_x[key] = jax_locations[:, [0], :]
            self.sensor_locations_y[key] = jax_locations[:, [1], :]
            self.sensor_locations_z[key] = jax_locations[:, [2], :]

    def extract_meteorology_information(self, meteorology_object):
        """Sub-function for extracting and storing meteorological information."""
        self.wind_speed = {}
        self.theta = {}
        self.wind_turbulence_horizontal = {}
        self.wind_turbulence_vertical = {}
        for key, meteo in meteorology_object.items():
            self.wind_speed[key] = jnp.array(meteo.wind_speed).reshape((meteo.wind_speed.shape[0], 1, 1))
            theta = np.arctan2(meteo.v_component, meteo.u_component)
            self.theta[key] = jnp.array(theta).reshape((meteo.wind_direction.shape[0], 1, 1))
            self.wind_turbulence_horizontal[key] = \
                jnp.array(meteo.wind_turbulence_horizontal).reshape((meteo.wind_turbulence_horizontal.shape[0], 1, 1))
            self.wind_turbulence_vertical[key] = \
                jnp.array(meteo.wind_turbulence_vertical).reshape((meteo.wind_turbulence_vertical.shape[0], 1, 1))

    def update_prefactors(self, state: dict, update_index: list = None) -> dict:
        """Update the coupling matrix based on the information in the state.

        Accounts for the situation where e.g. the source location or the wind sigma parameters change during the MCMC.

        TODO (14/06/24): There might be a better way to implement this (instead of a loop): but need to account for the
        fact that the point and beam sensor cases get handled differently (need a 3rd dimension for the beam knots).

        TODO (14/06/24): For the reversible jump case, might need to add an update of the number of sources and
        allocation vector.

        Args:
            state (dict): dictionary containing current state information.

        """
        if update_index is None:
            update_index = list(range(self.n_sources_max))
        else:
            update_index = [update_index]
        sensor_key_list = list(self.sensor_locations_x.keys())
        for idx in update_index:
            source_key = "z" + str(idx)
            coupling_key = "A" + str(idx)
            sensor_coupling_dict = {}
            source_x = jnp.atleast_3d(state[source_key][[0], :])
            source_y = jnp.atleast_3d(state[source_key][[1], :])
            source_z = jnp.atleast_3d(state[source_key][[2], :])
            for sensor_key in sensor_key_list:
                relative_x = self.sensor_locations_x[sensor_key] - source_x
                relative_y = self.sensor_locations_y[sensor_key] - source_y
                sensor_z = self.sensor_locations_z[sensor_key]
                coupling_array = compute_coupling_array_jax(
                    sensor_x=relative_x, sensor_y=relative_y, sensor_z=sensor_z, source_z=source_z,
                    wind_speed=self.wind_speed[sensor_key], theta=self.theta[sensor_key],
                    wind_turbulence_horizontal=self.wind_turbulence_horizontal[sensor_key],
                    wind_turbulence_vertical=self.wind_turbulence_vertical[sensor_key],
                    gas_density=self.gas_density
                )
                sensor_coupling_dict[sensor_key] = jnp.mean(coupling_array, axis=2)
            state[coupling_key] = jnp.concatenate(
                [sensor_coupling_dict[key] for key in sensor_key_list], axis=0
            )
        return state


@dataclass
class ScreenedManifoldMALA(ManifoldMALA):
    """Version of ManifoldMALA sampler which screens on the RJ on/off variable.

    If state["qi"] == 1 for i = 1,2,...,n, then a sample is generated for source i using the usual functionality.
    If state["qi"] == 0, then the source i is not sampled, and the coupling matrix is not updated for that source.
    """
    parameter_index: int = None

    def sample(self, current_state: dict) -> dict:
        """Overloaded version of the sample function which screens on the RJ on/off variable.

        Args:
            current_state (dict): The current state of the MCMC sampler.

        Returns:
            dict: The updated state after sampling. This is unchanged if state["qi"] == 0.

        """
        num_source = int(self.param[1:])
        if current_state["q"][num_source] == 1:
            return super().sample(current_state)
        return current_state

    def proposal(self, current_state: dict) -> Tuple[dict, np.ndarray, np.ndarray]:
        """Overloaded proposal."""
        # prop_state = deepcopy(current_state)
        prop_state = current_state.copy()
        prop_state[self.param] = jnp.copy(current_state[self.param])
        # TODO (04/07/25): does this work? And do we also need to copy the coupling?

        mu_cr, chol_cr = self._proposal_params(current_state)
        prop_state[self.param] = gmrf.sample_normal(mu_cr, L=chol_cr)
        # prop_state = self.model["y"].mean.update_prefactors(prop_state, update_index=self.parameter_index)
        _, prop_state = self.model["y"].log_p(prop_state, update_index=self.parameter_index)
        # TODO (17/06/25): don't need full density evaluation here, could save some computation by separately compiling
        # the coupling matrix. Check.
        logp_pr_g_cr = self._log_proposal_density(prop_state, mu_cr, chol_cr)

        mu_pr, chol_pr = self._proposal_params(prop_state)
        logp_cr_g_pr = self._log_proposal_density(current_state, mu_pr, chol_pr)

        return prop_state, logp_pr_g_cr, logp_cr_g_pr


@dataclass
class SourceReversibleJump(ReversibleJump):
    """Overloaded version of the ReversibleJump sampler, which handles the atomized source set up

    TODO (17/06/25): The reversible jump acceptance now involves

    """
    indicator_var: str = "q"
    source_variables: dict = None

    def __post_init__(self):
        super().__post_init__()
        self.source_variables = {"s" + str(i): "z" + str(i) for i in range(self.n_max)}
        self.other_variables = list(
            set(self.model.keys()) - set(self.source_variables.keys()) - set(self.source_variables.values())
        )

    def birth_proposal(self, current_state):
        """Overloaded."""

        prop_state = deepcopy(current_state)
        prop_state[self.param] += 1
        zero_loc = jnp.argwhere(prop_state[self.indicator_var].flatten() == 0)
        birth_index = int(zero_loc[0, 0])
        prop_state[self.indicator_var] = prop_state[self.indicator_var].at[birth_index].set(1)
        # prop_state[self.indicator_var][birth_index] = 1
        birth_location = "z" + str(birth_index)
        birth_rate = "s" + str(birth_index)

        # TODO (13/06/25): next section could be done with "associated parameter" or something.
        prop_state[birth_location] = self.model[birth_location].rvs(state=prop_state, n=1)
        prop_state[birth_rate] = self.model[birth_rate].rvs(state=prop_state, n=1)
        log_location_density, _ = self.model[birth_location].log_p(prop_state, by_observation=True)
        log_rate_density, _ = self.model[birth_rate].log_p(prop_state)
        log_prop_density = log_location_density + log_rate_density

        # update coupling matrix element
        # prop_state = self.model["y"].mean.update_prefactors(prop_state, update_index=birth_index)
        _, prop_state = self.model["y"].log_p(prop_state, update_index=birth_index)

        p_birth, p_death = self.get_move_probabilities(current_state, True)
        logp_pr_g_cr = np.log(p_birth) + log_prop_density[-1]
        logp_cr_g_pr = np.log(p_death)

        return prop_state, logp_pr_g_cr, logp_cr_g_pr

    def death_proposal(self, current_state):
        """Overloaded."""

        prop_state = deepcopy(current_state)
        prop_state[self.param] -= 1
        ones_loc = jnp.argwhere(prop_state[self.indicator_var].flatten() == 1)
        death_index = int(np.random.choice(np.array(ones_loc.flatten())))
        prop_state[self.indicator_var] = prop_state[self.indicator_var].at[death_index].set(0)
        # prop_state[self.indicator_var][death_index] = 0
        death_location = "z" + str(death_index)
        death_rate = "s" + str(death_index)

         # TODO (13/06/25): next section could be done with "associated parameter" or something.
        log_location_density, _ = self.model[death_location].log_p(prop_state, by_observation=True)
        log_rate_density, _ = self.model[death_rate].log_p(prop_state)
        log_prop_density = log_location_density + log_rate_density

        p_birth, p_death = self.get_move_probabilities(current_state, False)
        logp_pr_g_cr = np.log(p_death)
        logp_cr_g_pr = np.log(p_birth) + log_prop_density[-1]

        return prop_state, logp_pr_g_cr, logp_cr_g_pr

    def _accept_reject_proposal(self, current_state, prop_state, logp_pr_g_cr, logp_cr_g_pr):
        """Overloaded."""
        self.accept_rate.increment_proposal()
        logp_cs = 0
        logp_pr = 0
        ct = 0
        for rate, loc in self.source_variables.items():
            if current_state[self.indicator_var][ct] == 1:
                logp_cr_rate, _ = self.model[rate].log_p(current_state)
                logp_cr_loc, _ = self.model[loc].log_p(current_state)
                logp_cs += (logp_cr_rate + logp_cr_loc)
            if prop_state[self.indicator_var][ct] == 1:
                logp_pr_rate, _ = self.model[rate].log_p(prop_state)
                logp_pr_loc, _ = self.model[loc].log_p(prop_state)
                logp_pr += (logp_pr_rate + logp_pr_loc)
            ct += 1
        logp_cs_y, _ = self.model["y"].log_p(current_state)
        logp_pr_y, _ = self.model["y"].log_p(prop_state)
        logp_cs_rho, _ = self.model["n_src"].log_p(current_state)
        logp_pr_rho, _ = self.model["n_src"].log_p(prop_state)
        likelihood_diff = logp_pr_y - logp_cs_y
        logp_pr += logp_pr_rho
        logp_cs += logp_cs_rho
        # for var in self.other_variables:
        #     logp_cr_dist, _ = self.model[var].log_p(current_state)
        #     logp_cs += logp_cr_dist
        #     logp_pr_dist, _ = self.model[var].log_p(prop_state)
        #     logp_pr += logp_pr_dist
        log_accept = likelihood_diff + logp_pr + logp_cr_g_pr - (logp_cs + logp_pr_g_cr)

        if self.accept_proposal(log_accept):
            current_state = prop_state
            self.accept_rate.increment_accept()
        return current_state


@dataclass
class NullSampler(MCMCSampler):
    """Implementing null sampler, to enable stoarge of something that is not directly sampled."""

    def sample(self, current_state: dict) -> dict:
        """Overloaded sample function which does nothing."""
        return current_state
