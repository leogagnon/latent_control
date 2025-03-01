import gymnasium as gym
from data.hmm import CompositionalHMMDataset
import jax.numpy as jnp
import jax.random as jr
import jax
import numpy as np
from functools import partial
import tensorflow_probability.substrates.jax.distributions as tfd
import os
from collections import OrderedDict, deque, namedtuple
from typing import Iterator, List, Tuple
import numpy as np
import pandas as pd
import seaborn as sn
import torch
from IPython.core.display import display
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import CSVLogger
from torch import Tensor, nn
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset


class HMMEnv(gym.Env):
    """Batched GYM environment for MetaHMM envs with actions"""

    def __init__(
        self,
        metahmm: CompositionalHMMDataset,
        seed: int,
    ) -> None:
        super().__init__()
        assert metahmm.cfg.has_actions, 'MetaHMM has to have actions enabled'

        self.metahmm = metahmm
        self.seed = seed

        self.states: jnp.array
        self.indices: jnp.array

        self.generator = np.random.default_rng(seed)

    def reset(self, batch_size):

        # Set HMM indices
        hmm_dist = tfd.Categorical(
            probs=jnp.full(shape=len(self.metahmm), fill_value=1 / len(self.metahmm))
        )
        self.indices = hmm_dist.sample(
            batch_size,
            seed=jr.PRNGKey(
                self.generator.integers(
                    np.iinfo(np.int32).min, np.iinfo(np.int32).max, dtype=np.int32
                )
            ),
        )

        # Sample initial states
        init_prob_dist = tfd.Categorical(
            probs=jax.vmap(self.metahmm.get_startprobs)(self.indices)
        )
        self.states = init_prob_dist.sample(
            1,
            seed=jr.PRNGKey(
                self.generator.integers(
                    np.iinfo(np.int32).min, np.iinfo(np.int32).max, dtype=np.int32
                )
            ),
        )[0]

    # step in a cycle family
    @partial(jax.jit, static_argnames="self")
    def _step_in_cycle_family(self, family, hmm_index, state, key):

        # Add a self loop when node is not in family
        transition = self.metahmm.get_family_transition(hmm_index)[family]
        transition = jnp.fill_diagonal(
            transition, val=(1.0 - transition.sum(1)), inplace=False
        )

        next_state = tfd.Categorical(probs=transition[state]).sample(1, seed=key)[0]
        next_obs = tfd.Categorical(
            probs=self.metahmm.get_emission(hmm_index)[next_state]
        ).sample(1, seed=key)[0]
        return next_state, next_obs

    # step in a base cycle
    @partial(jax.jit, static_argnames="self")
    def _step_in_base_cycle(self, hmm_index, state, key):
        next_state = tfd.Categorical(
            probs=self.metahmm.get_base_transition(hmm_index)[state]
        ).sample(1, seed=key)[0]
        next_obs = tfd.Categorical(
            probs=self.metahmm.get_emission(hmm_index)[next_state]
        ).sample(1, seed=key)[0]
        return next_state, next_obs

    # step in-place
    @partial(jax.jit, static_argnames="self")
    def _step_inplace(self, hmm_index, state, key):
        next_state = jnp.array([state])[0]
        next_obs = tfd.Categorical(
            probs=self.metahmm.get_emission(hmm_index)[next_state]
        ).sample(1, seed=key)[0]
        return next_state, next_obs

    # step in the hmm
    @partial(jax.jit, static_argnames="self")
    def _step_default(self, hmm_index, state, key):
        next_state = tfd.Categorical(
            probs=self.metahmm.get_transition(hmm_index)[state]
        ).sample(1, seed=key)[0]
        next_obs = tfd.Categorical(
            probs=self.metahmm.get_emission(hmm_index)[next_state]
        ).sample(1, seed=key)[0]
        return next_state, next_obs

    @partial(jax.jit, static_argnames="self")
    def _step(self, states, actions, indices, keys):
        branches = [
            self._step_default,
            self._step_inplace,
            self._step_in_base_cycle,
        ] + [
            partial(self._step_in_cycle_family, i)
            for i in range(self.metahmm.cfg.cycle_families)
        ]
        return jax.vmap(jax.lax.switch, [0, None, 0, 0, 0])(
            actions, branches, indices, states, keys
        )

    def step(self, actions_probs: jnp.array):

        key = jr.PRNGKey(
                self.generator.integers(
                    np.iinfo(np.int32).min, np.iinfo(np.int32).max, dtype=np.int32
                )
            )
        
        action_key, env_key = jr.split(key, 2)

        actions = tfd.Categorical(probs=actions_probs).sample(1, seed=action_key)[0]

        keys = jr.split(
            env_key,
            num=len(actions),
        )

        self.states, obs = self._step(self.states, actions, self.indices, keys)

        return obs
