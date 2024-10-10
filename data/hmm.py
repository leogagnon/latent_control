from functools import partial
from math import gcd
from hmmlearn import hmm
import numpy as np
from scipy.special import softmax
import torch
from torch.utils.data import Dataset, Subset
from itertools import product
import lightning as L
from typing import *
from dataclasses import dataclass
from hmmlearn.base import _hmmc
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing.connection import Connection
import math
from torch.nn.utils.rnn import pad_sequence
from numpy.random._generator import Generator
import jax
from dynamax.hidden_markov_model import CategoricalHMM
from dynamax.hidden_markov_model.models.categorical_hmm import (
    ParamsCategoricalHMM,
    ParamsStandardHMMTransitions,
    ParamsCategoricalHMMEmissions,
    ParamsStandardHMMInitialState,
)
import jax.numpy as jnp
import jax.random as jr
from dynamax.hidden_markov_model.parallel_inference import (
    HMMPosteriorFiltered,
    _condition_on,
    FilterMessage,
    lax,
)
from jax.scipy.special import logsumexp
from omegaconf import MISSING
from torch2jax import t2j, j2t

import gc


@dataclass
class CompositionalHMMDatasetConfig:
    tag: Optional[str] = None
    n_states: int = 30
    n_obs: int = 60
    context_length: Tuple[int] = (200, 200)
    context_length_dist: str = "uniform"
    block_diag_mask: bool = False
    seed: int = 42
    base_cycles: int = 4
    base_directions: int = 2
    base_speeds: int = 3
    cycle_families: int = 4
    group_per_family: int = 2
    cycle_per_group: int = 3
    family_directions: int = 2
    family_speeds: int = 2
    emission_groups: int = 4
    emission_group_size: int = 2
    emission_shifts: int = 2
    emission_edge_per_node: int = 3
    emission_noise: float = 1e-5


def cycle_to_transmat(cycle: List[int], n_states: int) -> np.array:

    transition = np.zeros(shape=(n_states, n_states), dtype=np.int16)
    for i in range(len(cycle)):
        transition[
            cycle[i],
            cycle[(i + 1) % (len(cycle))],
        ] = 1.0
    return transition


def preferential_attachement_edges(
    states: np.array, obs: np.array, n: int, generator: Generator
) -> List[Tuple[int]]:

    states_ids = np.arange(len(states))
    obs_ids = np.arange(len(obs))
    obs_degree = np.zeros_like(obs_ids)

    # Add initial edges from each state to a random obs
    init_obs = generator.choice(obs_ids, size=len(states), replace=False)
    obs_degree[init_obs] = 1
    edges = [(states[i], obs[init_obs[i]]) for i in range(len(states))]

    # Iteratively add <n> edges, sampling state at random and obs at random weighted by degree
    for i in range(n):
        s_id = generator.choice(states_ids)
        o_id = generator.choice(obs_ids, p=softmax(obs_degree))

        obs_degree[o_id] += 1
        edges.append((states[s_id], obs[o_id]))

    return edges


class CompositionalHMMDataset(Dataset):
    def __init__(self, cfg: CompositionalHMMDatasetConfig) -> None:
        super().__init__()

        print("Initializing dataset...", end="")
        self.cfg = cfg
        self.generator = np.random.default_rng(cfg.seed)
        self.index_to_latent = jnp.array(
            self._make_index_to_latent(), device=jax.devices("cpu")[0]
        )
        self.latent_transmat = jnp.array(
            self._make_env_transition(), device=jax.devices("cpu")[0]
        )
        self.latent_emissions = jnp.array(
            self._make_env_emission(), device=jax.devices("cpu")[0]
        )
        print("Done!")

        self.hmm = CategoricalHMM(
            num_states=self.cfg.n_states, emission_dim=1, num_classes=self.cfg.n_obs
        )
        self.val_mode = False
        self.BOS_ID = self.cfg.n_obs
        self.PAD_ID = (
            -100
        )  # Because this is the default "ignore token" for cross entropy, TODO make this more future-proof

    @partial(jax.jit, static_argnames="self")
    def get_transition(self, index):
        latent = self.index_to_latent[index]
        transition_latent = latent[: (3 + self.cfg.cycle_families + 2)]
        return self.latent_transmat[tuple(transition_latent)]

    @partial(jax.jit, static_argnames="self")
    def get_emission(self, index):
        latent = self.index_to_latent[index]
        emission_latent = latent[-(self.cfg.emission_groups + 1) :]
        return self.latent_emissions[tuple(emission_latent)]

    @partial(jax.jit, static_argnames="self")
    def get_startprobs(self, index):
        return jnp.ones(self.cfg.n_states) / self.cfg.n_states

    @partial(jax.jit, static_argnames="self")
    def filter(self, index, X):
        r"""Filter algorithm

        Args:
            index: Underlying HMM
            X: Sequence of observation (WITHOUT BOS)

        Returns:
            log_p(x_{1..t} | alpha), p(z_t | x_{1...t}, alpha)

        """
        emission_matrix = self.get_emission(index)
        log_likelihoods = jnp.log(emission_matrix[:, X].T)
        initial_probs = self.get_startprobs(index)
        transition_matrix = self.get_transition(index)

        T, K = log_likelihoods.shape

        @jax.vmap
        def marginalize(m_ij, m_jk):
            A_ij_cond, lognorm = _condition_on(m_ij.A, m_jk.log_b)
            A_ik = A_ij_cond @ m_jk.A
            log_b_ik = m_ij.log_b + lognorm
            return FilterMessage(A=A_ik, log_b=log_b_ik)

        # Initialize the messages
        A0, log_b0 = _condition_on(initial_probs, log_likelihoods[0])
        A0 *= jnp.ones((K, K))
        log_b0 *= jnp.ones(K)
        A1T, log_b1T = jax.vmap(_condition_on, in_axes=(None, 0))(
            transition_matrix, log_likelihoods[1:]
        )
        initial_messages = FilterMessage(
            A=jnp.concatenate([A0[None, :, :], A1T]),
            log_b=jnp.vstack([log_b0, log_b1T]),
        )

        # Run the associative scan
        partial_messages = lax.associative_scan(marginalize, initial_messages)

        # Extract the marginal log likelihood and filtered probabilities
        log_like = jnp.concatenate([jnp.array([1.0]), partial_messages.log_b[:, 0]])
        z_post = jnp.concatenate([initial_probs[None], partial_messages.A[:, 0, :]])
        
        log_like = jnp.nan_to_num(log_like, nan=-jnp.inf)
        z_post = jnp.nan_to_num(z_post, nan=0.0)

        # Package into a posterior object
        return log_like, z_post

    @partial(jax.jit, static_argnames="self")
    def posterior_predictive(self, indices, X):
        
        # log_p(x_{1..t} | alpha)
        # p(z_t | x_{1...t}, alpha)
        log_like, z_post = jax.vmap(self.filter, (0, None))(indices, X)

        # p(x_{t+1} | x_{1...t}, alpha) = sum_z p(x_{t+1} | z_{t+1}, alpha) p(z_{t+1} | z_t, alpha) p(z_t | x_{1...t}, alpha)
        log_pp_given_alpha = jnp.log(
            jnp.einsum(
                "atz,azv,avx->atx",
                z_post,
                jax.vmap(self.get_transition)(indices),
                jax.vmap(self.get_emission)(indices),
            )
        )

        # p(alpha | x_{1...t}) = p(x_{1...t} | alpha) p(alpha) / sum_{alpha} p(x_{<t} | alpha) p(alpha)
        log_alpha_post = log_like - jnp.log(len(indices))
        log_alpha_post = log_alpha_post - logsumexp(log_alpha_post, axis=0)[None]

        # p(x_t | x_{<t}) = \sum_{alpha} p(x_t | x_{<t}, alpha) p(alpha | x_{<t})
        pp = jnp.nan_to_num(
            jnp.exp(logsumexp(log_pp_given_alpha + log_alpha_post[..., None], axis=0)),
            nan=0.0,
        )

        return pp

    def _make_env_transition(self):

        states = np.arange(self.cfg.n_states)

        # Generate base cycles
        base_transmat = np.zeros(
            shape=(
                self.cfg.base_cycles,
                self.cfg.base_directions,
                self.cfg.base_speeds,
                self.cfg.n_states,
                self.cfg.n_states,
            ),
            dtype=np.float16,
        )
        for i in range(self.cfg.base_cycles):
            # The base cycle is an ordering of all the nodes
            base_cycle = self.generator.permutation(np.arange(len(states)))
            for j in range(self.cfg.base_directions):
                # Potentially reverse the direction of the cycle
                flipped_cycle = np.flip(base_cycle) if (j == 1) else base_cycle
                for k in range(self.cfg.base_speeds):
                    # Potentially accelate the speed at which the cycle is traversed
                    speed = k + 1
                    if gcd(speed, len(flipped_cycle)) == 1:
                        speed_cycle = [
                            flipped_cycle[(speed * m) % len(flipped_cycle)]
                            for m in range(len(flipped_cycle))
                        ]
                        base_transmat[i, j, k] = cycle_to_transmat(
                            speed_cycle, self.cfg.n_states
                        )
                    # Potentially this creates multiple non-overlapping cycles
                    else:
                        for l in range(gcd(speed, len(flipped_cycle))):
                            speed_cycle = [
                                flipped_cycle[(speed * m + l) % len(flipped_cycle)]
                                for m in range(len(flipped_cycle) // speed)
                            ]
                            base_transmat[i, j, k] += cycle_to_transmat(
                                speed_cycle, self.cfg.n_states
                            )

        # Generate cycle families
        family_transmat = np.zeros(
            shape=(
                self.cfg.cycle_families,
                self.cfg.group_per_family,
                self.cfg.family_directions,
                self.cfg.family_speeds,
                self.cfg.n_states,
                self.cfg.n_states,
            ),
            dtype=np.float16,
        )
        for i in range(self.cfg.cycle_families):
            for j in range(self.cfg.group_per_family):
                # Generate a group of cycle
                group = [
                    self.generator.choice(states, size=length, replace=False)
                    for length in self.generator.integers(
                        3, 9, size=self.cfg.cycle_per_group
                    )
                ]
                for k in range(self.cfg.family_directions):
                    # Potentially flip all the cycles in the group
                    flipped_group = [np.flip(c) for c in group] if (k == 1) else group
                    for l in range(self.cfg.family_speeds):
                        # Potentially accelerate the speed at which all the cycles are traversed
                        speed = l + 1
                        for c in flipped_group:
                            if gcd(speed, len(c)) == 1:
                                speed_cycle = [
                                    c[(speed * m) % len(c)]
                                    for m in range(
                                        len(c) // speed
                                        if gcd(speed, len(c)) != 1
                                        else len(c)
                                    )
                                ]
                                family_transmat[i, j, k, l] = cycle_to_transmat(
                                    speed_cycle, self.cfg.n_states
                                )
                            else:
                                for n in range(gcd(speed, len(c))):
                                    speed_cycle = [
                                        flipped_cycle[
                                            (speed * m + n) % len(flipped_cycle)
                                        ]
                                        for m in range(len(flipped_cycle) // speed)
                                    ]
                                    family_transmat[i, j, k, l] += cycle_to_transmat(
                                        speed_cycle, self.cfg.n_states
                                    )

        latents = (
            [
                self.cfg.base_cycles,
                self.cfg.base_directions,
                self.cfg.base_speeds,
            ]
            + [self.cfg.group_per_family] * self.cfg.cycle_families
            + [self.cfg.family_directions, self.cfg.family_speeds]
        )
        latent_transitions = np.zeros(
            shape=latents + [self.cfg.n_states, self.cfg.n_states], dtype=np.float16
        )

        for latent in product(*[range(n) for n in latents]):
            base_id, base_direction, base_speed = latent[0], latent[1], latent[2]
            family_ids = latent[3 : (3 + self.cfg.cycle_families)]
            group_direction, group_speed = latent[-2], latent[-1]

            # Add relevant cycles
            cycles = [base_transmat[base_id, base_direction, base_speed]] + [
                family_transmat[i, group_id, group_direction, group_speed]
                for (i, group_id) in enumerate(family_ids)
            ]
            transmat = np.stack(cycles).sum(0)
            with np.errstate(divide="ignore", invalid="ignore"):
                transmat = transmat / transmat.sum(1)[:, None]
                transmat = np.nan_to_num(transmat, nan=0.0)

            latent_transitions[tuple(latent)] = transmat

        return latent_transitions

    def _make_env_emission(self):

        states = np.arange(self.cfg.n_states)
        obs = np.arange(self.cfg.n_obs)

        state_groups = np.array_split(states, self.cfg.emission_groups)
        emissions = np.zeros(
            shape=(
                self.cfg.emission_groups,
                self.cfg.emission_group_size,
                self.cfg.emission_shifts,
                self.cfg.n_states,
                self.cfg.n_obs,
            )
        )
        for i in range(self.cfg.emission_groups):
            group = list(state_groups[i])
            for j in range(self.cfg.emission_group_size):
                edges = preferential_attachement_edges(
                    group,
                    obs,
                    self.cfg.emission_edge_per_node * len(group),
                    generator=self.generator,
                )
                for k in range(self.cfg.emission_shifts):
                    # Shift starting edge within
                    for l in range(len(edges)):
                        source_idx = group.index(edges[l][0])
                        shifted_edge = (
                            group[(source_idx + k) % len(group)],
                            edges[l][1],
                        )
                        emissions[(i, j, k) + shifted_edge] = 1

        latents = [self.cfg.emission_group_size] * self.cfg.emission_groups + [
            self.cfg.emission_shifts
        ]
        latent_emissions = np.zeros(
            shape=latents + [self.cfg.n_states, self.cfg.n_obs], dtype=np.float16
        )
        for latent in product(*[range(n) for n in latents]):
            groups_id = latent[: self.cfg.emission_groups]
            emission_shift = latent[-1]

            emi = np.stack(
                [
                    emissions[group_id, emissions_id, emission_shift]
                    for (group_id, emissions_id) in enumerate(groups_id)
                ]
            ).sum(0)

            # Add a bit of noise
            emi = emi + self.cfg.emission_noise

            # Normalize
            with np.errstate(divide="ignore", invalid="ignore"):
                emi = emi / emi.sum(1)[:, None]
                emi = np.nan_to_num(emi, nan=0.0)

            latent_emissions[tuple(latent)] = emi

        return latent_emissions

    def _make_index_to_latent(self):

        latents = (
            [
                self.cfg.base_cycles,
                self.cfg.base_directions,
                self.cfg.base_speeds,
            ]
            + [self.cfg.group_per_family] * self.cfg.cycle_families
            + [self.cfg.family_directions, self.cfg.family_speeds]
            + [self.cfg.emission_group_size] * self.cfg.emission_groups
            + [self.cfg.emission_shifts]
        )
        index_to_latent = list(product(*[range(n) for n in latents]))
        index_to_latent = np.array(index_to_latent, dtype=np.int16)

        return index_to_latent

    def __len__(self):
        return len(self.index_to_latent)

    @partial(jax.jit, static_argnames=["self", "n_steps"])
    def sample(self, index, n_steps, key):
        params = ParamsCategoricalHMM(
            initial=ParamsStandardHMMInitialState(self.get_startprobs(index)),
            transitions=ParamsStandardHMMTransitions(self.get_transition(index)),
            emissions=ParamsCategoricalHMMEmissions(
                jnp.reshape(
                    self.get_emission(index),
                    shape=(self.cfg.n_states, 1, self.cfg.n_obs),
                )
            ),
        )

        # We generate for n_steps - 1 because we add the BOS token
        Z, X = self.hmm.sample(params, key, n_steps - 1)
        return jnp.concatenate([jnp.array([self.BOS_ID]), X[:, 0]])

    def __getitems__(
        self,
        indices: List[int],
        seed: Optional[int] = None,
        n_steps: Optional[int] = None,
    ):

        indices = jnp.array(indices)
        batch_size = len(indices)

        variable_len = (self.cfg.context_length[0] != self.cfg.context_length[1]) & (
            not self.val_mode
        )

        if seed is None:
            seed = self.generator.integers(0, 1e10)

        if n_steps is None:
            n_steps = self.cfg.context_length[1]
        else:
            variable_len = False

        # Generate sequences in parallel with jax, then convert to torch
        seqs = jax.vmap(self.sample, (0, None, 0))(
            indices,
            n_steps,
            jr.split(jr.PRNGKey(seed), batch_size),
        )
        seqs = j2t(seqs)
        attn_masks = None
        pad_masks = None

        # Potentially generate attention masks, or mask suffixes
        if variable_len:

            if self.cfg.block_diag_mask:
                # Basic causal masking
                attn_masks = (
                    torch.tril(
                        torch.ones(
                            1, self.cfg.context_length[1], self.cfg.context_length[1]
                        ),
                        diagonal=0,
                    )
                    .tile(batch_size, 1, 1)
                    .to(torch.bool)
                )

                # Block-diagonal masking
                for i in range(batch_size):
                    idx = self.generator.integers(
                        self.cfg.context_length[0],
                        self.cfg.context_length[1] + 1,
                        self.cfg.context_length[1] // 2,
                    )
                    idx = idx[np.cumsum(idx) < self.cfg.context_length[1]].tolist()
                    idx = idx + [self.cfg.context_length[1] - sum(idx)]
                    attn_masks[i] *= torch.block_diag(
                        *[torch.ones((l, l), dtype=torch.bool) for l in idx]
                    )
                attn_masks = attn_masks.unsqueeze(1)
            else:
                # Simply replace a random suffix length with padding
                pad_masks = (
                    torch.arange(self.cfg.context_length[1]).tile(batch_size, 1)
                    > torch.Tensor(
                        self.generator.integers(
                            self.cfg.context_length[0],
                            self.cfg.context_length[1] + 1,
                            batch_size,
                        )
                    )[:, None]
                )

        return (seqs, attn_masks, pad_masks)
