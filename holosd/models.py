"""Flax building blocks for HoloSD's SAC control (Phase B).

Only the reusable SAC pieces live here -- an ``MLP``, a twin Q-critic, and
tanh-Gaussian action sampling. The chart network ``Psi`` and the oscillator
exploration actor are defined directly in ``holosd_final``.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn

LOG_STD_MIN = -10.0
LOG_STD_MAX = 2.0


class MLP(nn.Module):
    hidden: int
    n_layers: int
    out_dim: int

    @nn.compact
    def __call__(self, x):
        for _ in range(self.n_layers):
            x = nn.relu(nn.Dense(self.hidden)(x))
        return nn.Dense(self.out_dim)(x)


def sample_action(mean, log_std, key):
    """Reparameterised tanh-Gaussian sample + log prob (SAC)."""
    std = jnp.exp(log_std)
    eps = jax.random.normal(key, mean.shape)
    pre = mean + std * eps
    a = jnp.tanh(pre)
    # log prob with tanh correction
    log_prob = (-0.5 * (eps ** 2) - log_std - 0.5 * jnp.log(2 * jnp.pi)).sum(-1)
    log_prob -= jnp.log(1.0 - a ** 2 + 1e-6).sum(-1)
    return a, log_prob


class DoubleCritic(nn.Module):
    """Twin Q(obs, action)."""
    hidden: int = 1024
    n_layers: int = 2

    @nn.compact
    def __call__(self, obs, action):
        x = jnp.concatenate([obs, action], axis=-1)
        q1 = MLP(self.hidden, self.n_layers, 1)(x)
        q2 = MLP(self.hidden, self.n_layers, 1)(x)
        return q1.squeeze(-1), q2.squeeze(-1)
