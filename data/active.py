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
from torch2jax import j2t, t2j


class HMMEnv(gym.Env):
    """Batched GYM environment for MetaHMM envs with actions"""

    def __init__(
        self,
        metahmm: CompositionalHMMDataset,
        seed: int,
    ) -> None:
        super().__init__()
        assert metahmm.cfg.has_actions, "MetaHMM has to have actions enabled"

        self.metahmm = metahmm
        self.seed = seed

        self.states: jnp.array
        self.indices: jnp.array
        self.__history: list

        self.generator = np.random.default_rng(seed)

    @property
    def history(self):
        if hasattr(self, "_HMMEnv__history"):
            return j2t(jnp.concatenate(self.__history, axis=-1))
        else:
            return None

    def reset(self, batch_size):

        key = jr.PRNGKey(
            self.generator.integers(
                np.iinfo(np.int32).min, np.iinfo(np.int32).max, dtype=np.int32
            )
        )
        hmm_key, init_state_key, init_obs_key = jr.split(key, 3)

        # Set HMM indices
        hmm_dist = tfd.Categorical(
            probs=jnp.full(shape=len(self.metahmm), fill_value=1 / len(self.metahmm))
        )
        self.indices = hmm_dist.sample(
            batch_size,
            seed=hmm_key,
        )

        # Sample initial states
        self.states = jax.vmap(self._sample_initstate, [0, 0])(
            self.indices, jr.split(init_state_key, len(self.indices))
        )
        obs = jax.vmap(self._sample_obs, [0, 0, 0])(
            self.states, self.indices, jr.split(init_obs_key, len(self.indices))
        )[:, None]

        self.__history = [obs]

    @partial(jax.jit, static_argnames="self")
    def _sample_obs(self, state, hmm_index, key):
        return tfd.Categorical(
            probs=self.metahmm.get_emission(hmm_index)[state]
        ).sample(1, seed=key)[0]

    @partial(jax.jit, static_argnames="self")
    def _sample_initstate(self, hmm_index, key):
        return tfd.Categorical(probs=self.metahmm.get_startprobs(hmm_index)).sample(
            1,
            seed=key,
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

    def step(self, action_probs):
        assert action_probs.shape[-1] == len(self.metahmm.ACTIONS)
        
        if isinstance(action_probs, torch.Tensor):
            action_probs = t2j(action_probs)

        key = jr.PRNGKey(
            self.generator.integers(
                np.iinfo(np.int32).min, np.iinfo(np.int32).max, dtype=np.int32
            )
        )

        action_key, env_key = jr.split(key, 2)

        actions = tfd.Categorical(probs=action_probs).sample(1, seed=action_key)[0]

        keys = jr.split(
            env_key,
            num=len(actions),
        )

        self.states, obs = self._step(self.states, actions, self.indices, keys)

        self.__history.append(jnp.stack([actions, obs], axis=-1))

        return obs


PPOBatch = namedtuple(
    "PPOBatch", ["sequences", "advantages", "returns", "values", "logprobs", "entropy"]
)


class PPODataset(Dataset):
    def __init__(
        self, sequences, advantages, returns, values, logprobs, entropy, repeats=1
    ) -> None:
        super().__init__()

        self.sequences = sequences
        self.advantages = advantages
        self.returns = returns
        self.values = values
        self.logprobs = logprobs
        self.entropy = entropy

        self.repeats = repeats

        assert [
            len(sequences) == len(val)
            for val in [advantages, returns, values, logprobs, entropy]
        ], "All tensors should be the same length as <sequences>"

    def __len__(self):
        return int(len(self.sequences) * self.repeats)

    def __getitems__(self, indices):
        indices = torch.LongTensor(indices) % len(self.sequences)
        return PPOBatch(
            sequences=self.sequences[indices],
            advantages=self.advantages[indices],
            returns=self.returns[indices],
            values=self.values[indices],
            logprobs=self.logprobs[indices],
            entropy=self.entropy[indices],
        )
