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

from helper_functions import generate_pyelq_test_model

from openmcmc.parameter import Identity
from openmcmc.distribution.distribution_jax import Normal_jax
from openmcmc.model import Model
from openmcmc.sampler.sampler import NormalNormal
from openmcmc.sampler.metropolis_hastings import ManifoldMALA
from openmcmc.mcmc import MCMC
from pyelq.component.source_model import SourceModelParameter

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
model.components["source"].n_sources_max = 3

# populate sources
for i in range(model.components["source"].n_sources_max):
    state["z" + str(i)] = jnp.array(
        np.random.uniform(low=np.array([[0], [0], [0]]), high=np.array([[30], [30], [5]]), size=(3, 1))
    )

# populate emission rates
for i in range(model.components["source"].n_sources_max):
    state["s" + str(i)] = jnp.ones(shape=(1, 1))

# populate coupling matrix
n_data = 1000
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
state["Q"] = (100.0) * jnp.eye(state["y"].size)

# create the data likelihood
likelihood_y = Normal_jax(
    response="y",
    grad_list=["z" + str(i) for i in range(model.components["source"].n_sources_max)] + \
        ["s" + str(i) for i in range(model.components["source"].n_sources_max)],
    mean=source_parameter,
    precision=Identity("Q"),
    update_index_max=model.components["source"].n_sources_max,
    jit_compile=True
)
likelihood_y.param_list = ["s" + str(i) for i in range(model.components["source"].n_sources_max)] + \
    ["z" + str(i) for i in range(model.components["source"].n_sources_max)]

# test the data likelihood
log_p, state = likelihood_y.log_p(state)

"""Set up the rest of the MCMC sampler components."""

model_list = [likelihood_y]
for i in range(model.components["source"].n_sources_max):
    # prior state stuff
    state["mu_s" + str(i)] = 0.0
    state["P_s" + str(i)] = 1.0 / jnp.power(100.0, 2)
    # prior distribution stuff
    model_list.append(
        Normal_jax(
            response="s" + str(i),
            grad_list=["s" + str(i)],
            mean=Identity("mu_s" + str(i)),
            precision=Identity("P_s" + str(i)),
            update_index_max=model.components["source"].n_sources_max,
            jit_compile=True
        )
    )
    model_list[-1].param_list = ["s" + str(i)]
mdl = Model(model_list)

sampler_list = []
for i in range(model.components["source"].n_sources_max):
    sampler_list.append(ManifoldMALA("z" + str(i), mdl, step=0.01))
    sampler_list.append(NormalNormal("s" + str(i), mdl))

initial_state = deepcopy(state)
mcmc = MCMC(initial_state, sampler_list, model=mdl, n_burn=10, n_iter=500)
mcmc.run_mcmc()

# NOTE: scaled identity added to the precision matrix in the mMALA sampler, to make it more stable. Should introduce
# proper priors for the source locations and remove this.

# NOTE: