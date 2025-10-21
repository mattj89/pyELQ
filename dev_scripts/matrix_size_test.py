"""
Test without using any of the low-level pyELQ code whether we can change the size of a coupling matrix inside the JAX
code.
"""

from functools import partial
import numpy as np
import jax.numpy as jnp
from jax import jit, grad

from tqdm import tqdm

def make_plume(sensor_x, sensor_y, source_x, source_y, wind_speed, theta):
    """Simple 2D plume function."""
    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)

    delta_x = sensor_x - source_x
    delta_y = sensor_y - source_y

    distance_x = cos_theta * delta_x + sin_theta * delta_y
    distance_y = -sin_theta * delta_x + cos_theta * delta_y

    sigma_hor = jnp.tan(10.0 * (jnp.pi / 180)) * jnp.abs(distance_x)

    plume_coupling = (
        (1 / (2 * jnp.pi * wind_speed * sigma_hor))
        * jnp.exp(-0.5 * (distance_y / sigma_hor) ** 2)
    )
    plume_coupling = jnp.divide(plume_coupling * 1e6, (0.67 * 3600))
    plume_coupling = jnp.where(
        jnp.logical_or(distance_x < 0, plume_coupling < 1e-6), 0, plume_coupling
    )
    return plume_coupling


def likelihood(state: dict, grad_value: jnp.ndarray = None, grad_name: str = None) -> jnp.ndarray:
    """Normal likelihood function which calculates the plume coupling under the surface."""
    if grad_value is not None:
        state_copy = state.copy()
        state_copy[grad_name] = grad_value
        state = state_copy
    state["A"] = make_plume(
        sensor_x=state["sensor_x"],
        sensor_y=state["sensor_y"],
        source_x=state["z"][[0], :],
        source_y=state["z"][[1], :],
        wind_speed=state["wind_speed"],
        theta=state["wind_direction"]
    )
    residual = state["y"] - state["A"] @ state["s"]
    log_likelihood = -0.5 * jnp.vdot(residual, residual) * state["precision"]
    return log_likelihood, state


# set up the jit compiled versions
likelihood_jit = jit(likelihood, static_argnums=(1, 2))
grad_likelihood = {}
for param in ["s", "z"]:
    def temp_lik(state: dict, grad_value: jnp.ndarray, grad_name: str = param) -> jnp.ndarray:
        log_lik, _ = likelihood(state, grad_value=grad_value, grad_name=grad_name)
        return log_lik
    grad_likelihood[param] = jit(grad(temp_lik, argnums=1))

"""
Run and speed test.
"""

# numbers
num_tests = 10
num_replicates = 1000
num_sensors = 9
num_data = 5000

# create variables
sensor_x = jnp.ones((num_data, 1))
sensor_y = jnp.ones((num_data, 1))
wind_speed = jnp.ones((num_data, 1))
theta = jnp.ones((num_data, 1))
y = jnp.zeros((num_data, 1))
precision = 1.0
state = {
    "sensor_x": sensor_x,
    "sensor_y": sensor_y,
    "wind_speed": wind_speed,
    "wind_direction": theta,
    "y": y,
    "precision": precision
}

for test in range(num_tests):
    # update the source array
    state["z"] = jnp.zeros(shape=(2, test + 1))
    state["s"] = jnp.ones(shape=(test + 1, 1))
    print("Running with {} sources.".format(test + 1))
    for rep in tqdm(range(num_replicates)):
        # evaluate the log-likelihood
        log_lik, state = likelihood_jit(state)
        # evaluate the gradient
        for param in ["s", "z"]:
            grad_lik = grad_likelihood[param](state, state[param])


"""
Run where the number of sources changes each iteration
"""

print("Testing when size continually varies.")
for test in tqdm(range(num_replicates * num_tests)):
    random_size = np.random.randint(1, 11)
    state["z"] = jnp.zeros(shape=(2, random_size))
    state["s"] = jnp.ones(shape=(random_size, 1))
    # evaluate the log-likelihood
    log_lik, state = likelihood_jit(state)
    # evaluate the gradient
    for param in ["s", "z"]:
        grad_lik = grad_likelihood[param](state, state[param])

"""
NOTE: no issues with this- so i think we can just go back to the version where we change the size of the matrix each
time based on the number of sources.

"""