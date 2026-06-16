"""Kinematic Reeds-Shepp car environment for HoloSD (JAX).

State ``s = [x, y, theta, phi]`` (position, heading, steering angle), control
``a = [v_norm, dphi_norm]`` (signed speed, steering rate); poses are integrated by
exact-arc updates, so the minimum turning radius is ``R_min = wheelbase / tan(phi_max)``.

This module holds only the dynamics the final algorithm uses; the rollout, render,
and 2-D LiDAR helpers from the archived experiment lineage have been removed.
"""

from functools import partial
from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class EnvParams:
    dt: float = 0.1
    v_max: float = 1.0
    wheelbase: float = 0.5
    phi_max: float = 0.6
    dphi_max: float = 1.5
    max_path_length: int = 200


STATE_DIM = 4    # [x, y, theta, phi]
ACT_DIM = 2      # [v_norm, dphi_norm]


@partial(jax.jit, static_argnums=(2,))
def env_step(state, action, p):
    """Exact-arc kinematic step. state [...,4], action [...,2] -> next state [...,4]."""
    action = jnp.clip(action, -1.0, 1.0)
    v = p.v_max * action[..., 0]
    dphi = p.dphi_max * action[..., 1]
    x, y, theta, phi = state[..., 0], state[..., 1], state[..., 2], state[..., 3]
    phi = jnp.clip(phi + dphi * p.dt, -p.phi_max, p.phi_max)
    kappa = jnp.tan(phi) / p.wheelbase
    omega = v * kappa
    small = jnp.abs(omega) < 1e-8
    x_s = x + v * p.dt * jnp.cos(theta)
    y_s = y + v * p.dt * jnp.sin(theta)
    so = jnp.where(small, 1.0, omega)
    x_a = x + (v / so) * (jnp.sin(theta + omega * p.dt) - jnp.sin(theta))
    y_a = y - (v / so) * (jnp.cos(theta + omega * p.dt) - jnp.cos(theta))
    x_new = jnp.where(small, x_s, x_a)
    y_new = jnp.where(small, y_s, y_a)
    theta_new = theta + omega * p.dt
    return jnp.stack([x_new, y_new, theta_new, phi], axis=-1)
