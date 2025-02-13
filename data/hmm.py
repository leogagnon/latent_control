import math
import multiprocessing as mp
from dataclasses import dataclass
from functools import partial
from itertools import product
from math import gcd
from multiprocessing.connection import Connection
from typing import *

import jax
import jax.numpy as jnp
import jax.random as jr
import lightning as L
import numpy as np
import torch
from dynamax.hidden_markov_model import CategoricalHMM
from dynamax.hidden_markov_model.models.categorical_hmm import (
    ParamsCategoricalHMM, ParamsCategoricalHMMEmissions,
    ParamsStandardHMMInitialState, ParamsStandardHMMTransitions)
from dynamax.hidden_markov_model.parallel_inference import (
    FilterMessage, HMMPosteriorFiltered, _condition_on, lax)
from jax.scipy.special import logsumexp
from numpy.random._generator import Generator
from omegaconf import MISSING
from scipy.special import softmax
from torch2jax import j2t, t2j
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, Subset
from tqdm import tqdm


@dataclass
class CompositionalHMMDatasetConfig:
    tag: Optional[str] = None
    n_states: int = 30
    n_obs: int = 60
    context_length: Tuple[int] = (200, 200)
    context_length_dist: str = "uniform"
    adjust_varlen_batch: bool = False
    start_at_n: Optional[int] = None
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

    def to_device(self, device):
        self.index_to_latent = jax.device_put(
            self.index_to_latent, jax.devices(device)[0]
        )
        self.latent_transmat = jax.device_put(
            self.latent_transmat, jax.devices(device)[0]
        )
        self.latent_emissions = jax.device_put(
            self.latent_emissions, jax.devices(device)[0]
        )

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
    def filter(self, index, X, init=None):
        r"""Filter algorithm

        Args:
            index: Underlying HMM
            X: Sequence of observation

        Returns:
            log_likelihood: log_p(x_{1..t} | alpha), t \in [0,T]
            posterior: p(z_t | x_{1...t}, alpha), t \in [0,T] (NOTE: Includes p(z_0))

        """
        # Default initial state to <self.get_startprobs(index)>
        # initial_probs = jax.lax.select(
        #    initial_probs == False,
        #    on_true=self.get_startprobs(index),
        #    on_false=jnp.zeros(self.cfg.n_states).at[:].set(initial_probs),
        # )
        initial_probs = self.get_startprobs(index)
        emission_matrix = self.get_emission(index)
        log_likelihoods = jnp.log(emission_matrix[:, X].T)
        transition_matrix = self.get_transition(index)

        T, K = log_likelihoods.shape

        @jax.vmap
        def marginalize(m_ij, m_jk):
            A_ij_cond, lognorm = _condition_on(m_ij.A, m_jk.log_b)
            A_ik = A_ij_cond @ m_jk.A
            log_b_ik = m_ij.log_b + lognorm
            return FilterMessage(A=A_ik, log_b=log_b_ik)

        # Build initial messages
        if init is None:
            A0, log_b0 = _condition_on(initial_probs, log_likelihoods[0])
            A0 *= jnp.ones((K, K))
            log_b0 *= jnp.ones(K)
            A1T, log_b1T = jax.vmap(_condition_on, in_axes=(None, 0))(
                transition_matrix, log_likelihoods[1:]
            )
        else:
            A0, log_b0 = init[:-K], init[-K:]
            A0 = A0.reshape(K, K)

            A1T, log_b1T = jax.vmap(_condition_on, in_axes=(None, 0))(
                transition_matrix, log_likelihoods[0:]
            )
        initial_messages = FilterMessage(
            A=jnp.concatenate([A0[None, :, :], A1T]),
            log_b=jnp.vstack([log_b0, log_b1T]),
        )

        # Run the associative scan
        partial_messages = lax.associative_scan(marginalize, initial_messages)

        # Extract the marginal log likelihood and filtered probabilities (add p(z_0), p(x_0)=1)
        log_like = partial_messages.log_b[:, 0]
        z_post = partial_messages.A[:, 0, :]
        if init is None:
            log_like = jnp.concatenate([jnp.log(jnp.array([1.0])), log_like])
            z_post = jnp.concatenate([initial_probs[None], z_post])

        log_like = jnp.nan_to_num(log_like, nan=-jnp.inf)
        z_post = jnp.nan_to_num(z_post, nan=0.0)

        partial_messages = jnp.concatenate(
            [
                partial_messages.A[-T:].reshape(T, -1),
                partial_messages.log_b[-T:].reshape(T, -1),
            ],
            -1,
        )

        return log_like, z_post, partial_messages

    @partial(jax.jit, static_argnames="self")
    def bayesian_oracle(
        self, indices, X, initial_messages=False, log_alpha_prior=False
    ):
        """Posterior predictive

        Args:
            indices (jnp.array): Environments considered possible
            X (jnp.array): Sequence of observations

        Returns:
            posterior_predictive: p(x_t | x_{<t}), t \in [1, T+1] (NOTE: INCLUDES p(x_1))
            posterior_latent: p(z_t | x)
        """

        # log_p(x_{1..t} | alpha)
        # p(z_t | x_{1...t}, alpha)
        if initial_messages is False:
            log_x_given_alpha, z_given_x_alpha, messages = jax.vmap(
                self.filter, (0, None)
            )(indices, X)
        else:
            log_x_given_alpha, z_given_x_alpha, messages = jax.vmap(
                self.filter, (0, None, 0)
            )(indices, X, initial_messages)

        # p(x_{t+1} | x_{1...t}, alpha) = sum_z p(x_{t+1} | z_{t+1}, alpha) p(z_{t+1} | z_t, alpha) p(z_t | x_{1...t}, alpha)
        log_x_given_x_alpha = jnp.log(
            jnp.einsum(
                "atz,azv,avx->atx",
                z_given_x_alpha,
                jax.vmap(self.get_transition)(indices),
                jax.vmap(self.get_emission)(indices),
            )
        )

        # Compute p(alpha)
        log_alpha = jnp.full(
            shape=(len(indices), 1), fill_value=jnp.log(1 / len(indices))
        ) * jnp.any(log_alpha_prior == False) + jnp.zeros((len(indices), 1)).at[
            :, 0
        ].set(
            log_alpha_prior
        ) * jnp.any(
            log_alpha_prior != False
        )
        # p(alpha | x_{1...t}) = p(x_{1...t} | alpha) p(alpha) / sum_{alpha} p(x_{<t} | alpha) p(alpha)
        log_alpha_given_x = log_x_given_alpha + jnp.broadcast_to(
            log_alpha, log_x_given_alpha.shape
        )
        log_alpha_given_x = (
            log_alpha_given_x - logsumexp(log_alpha_given_x, axis=0)[None]
        )

        # p(x_{t+1} | x_{1...t}) = \sum_{alpha} p(x_{t+1} | x_{1...t}, alpha) p(alpha | x_{1...t})
        x_given_x = jnp.nan_to_num(
            jnp.exp(
                logsumexp(log_x_given_x_alpha + log_alpha_given_x[..., None], axis=0)
            ),
            nan=0.0,
        )

        # p(z_t | x_{1...t}) = \sum_{alpha} p(z_t | x_{1...t}, alpha) p(alpha | x_{1...t})
        z_given_x = jnp.nan_to_num(
            jnp.exp(
                logsumexp(
                    jnp.log(z_given_x_alpha) + log_alpha_given_x[..., None], axis=0
                )
            ),
            nan=0.0,
        )

        return {
            "post_pred": x_given_x,
            "z_post": z_given_x,
            "log_alpha_post": log_alpha_given_x.T,
            "messages": messages.transpose(1, 0, 2),
        }

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

    def get_latents_shape(self):
        latents_shape = (
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
        return latents_shape

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

    @property
    def latent_shape(self):
        return (
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

    @partial(jax.jit, static_argnames=["self", "n_steps", "reverse"])
    def sample(self, index, n_steps, key, initial_state=None, reverse=False):
        """Sample a sequence of observation from HMM <index>

        Args:
            index (int): ID of the HMM
            n_steps (int): Size of the sequence
            key (jr.PRNGKey): Jax randomness
            initial_state (int, optional): Starting state. Defaults to uniform over all states.
            reverse (bool, optional): Whether to sample in reverse (starting from the end, transition matrix transposed)

        Returns:
            X: Observations, jnp.array of size <n_steps>
            Y: States, jnp.array of size <n_steps>
        """
        startprobs = jax.lax.select(
            initial_state is not None,
            on_true=jax.nn.one_hot(initial_state, self.hmm.num_states),
            on_false=self.get_startprobs(index),
        )

        transitions = (
            self.get_transition(index).T if reverse else self.get_transition(index)
        )

        params = ParamsCategoricalHMM(
            initial=ParamsStandardHMMInitialState(startprobs),
            transitions=ParamsStandardHMMTransitions(transitions),
            emissions=ParamsCategoricalHMMEmissions(
                jnp.reshape(
                    self.get_emission(index),
                    shape=(self.cfg.n_states, 1, self.cfg.n_obs),
                )
            ),
        )

        Z, X = self.hmm.sample(params, key, n_steps)
        X = X[:, 0]

        if reverse:
            return jnp.flip(X), jnp.flip(Z)

        return X, Z

    def __getitem__(self, env):
        return self.__getitems__([env])

    def __getitems__(
        self,
        envs: List[int],
        seed: Optional[int] = None,
        length: Optional[Union[Tuple[int], int]] = None,
        intv_idx: Optional[Union[Tuple[int], int]] = None,
        intv_envs: Optional[List[int]] = None,
    ):
        assert ~np.logical_xor(
            intv_idx is None, intv_envs is None
        ), "prefix_len and prefix_indices should both be set or neither be set"

        envs = jnp.array(envs)
        batch_size = len(envs)
        intv_envs = jnp.array(intv_envs)

        out_dict = {}

        # Set length if not set
        if length is None:
            length = self.cfg.context_length
        if isinstance(length, int):
            length = (length, length)

        variable_len = (length[0] != length[1]) & (not self.val_mode)

        # Set seed if not set
        if seed is None:
            seed = self.generator.integers(0, 1e10)

        # Sample sequences of maximum length
        seqs, states = jax.vmap(self.sample, (0, None, 0))(
            envs,
            length[1],
            jr.split(jr.PRNGKey(seed), batch_size),
        )

        seqs, states = j2t(seqs), j2t(states)

        out_dict.update({"input_ids": seqs, "states": states, "envs": envs})

        # Mid sequence intervention
        if intv_idx is not None:
            assert (
                variable_len is False
            ), "Cannot use variable lenght with interventions"
            # intv_idx: timestep in the sequence where the first intervened transition happens
            if isinstance(intv_idx, Iterable):
                intv_idx = self.generator.integers(
                    low=intv_idx[0], high=intv_idx[1], size=batch_size
                )
            else:
                intv_idx = np.full(shape=(batch_size,), fill_value=intv_idx)

            # Compute the state of the HMM at timestep <intv_idx -1>
            start_states = states[torch.arange(batch_size), intv_idx - 1]

            # Simulate the intervened HMM starting from this state
            # NOTE: Important to change the seed here (hence the +1), else the network cheats
            seqs_intv, states_intv = jax.vmap(self.sample, (0, None, 0, 0))(
                intv_envs,
                length if length is not None else self.cfg.context_length[1],
                jr.split(jr.PRNGKey(seed + 1), batch_size),
                t2j(start_states),
            )
            # We remove the first observation because it came from state <intv_idx-1>
            seqs_intv, states_intv = j2t(seqs_intv)[:, 1:], j2t(states_intv)[:, 1:]

            raw_seqs = seqs.clone()
            raw_states = states.clone()

            for j in range(batch_size):
                # Intervene on the sequence
                seqs[j, intv_idx[j] :] = seqs_intv[j, : (seqs.shape[1] - intv_idx[j])]
                states[j, intv_idx[j] :] = states_intv[
                    j, : (states.shape[1] - intv_idx[j])
                ]

            # Generate masks so that we only train on the intervened trajectory (>= intv_idx)
            # Masks where True
            ignore_mask = (
                torch.arange(seqs.shape[1]).tile(len(seqs), 1)
                < torch.Tensor(intv_idx)[:, None]
            )

            out_dict.update(
                {
                    "input_ids": seqs,
                    "states": states,
                    "raw_seqs": raw_seqs,
                    "raw_states": raw_states,
                    "ignore_mask": ignore_mask,
                }
            )

            return out_dict

        if variable_len:
            if self.cfg.adjust_varlen_batch:
                seqlens_ = self.generator.integers(
                    length[0],
                    length[1] + 1,
                    5 * batch_size,
                ).tolist()
                cu_seqlens_ = np.cumsum(seqlens_)
                seqlens = []
                for i in range(batch_size):
                    n_seqs = int(np.sum(cu_seqlens_ <= length[1]))
                    seqlens.append(seqlens_[:n_seqs])
                    if sum(seqlens[-1]) != length[1]:
                        seqlens[-1].append(
                            length[1] - np.sum(cu_seqlens_[n_seqs - 1]).item()
                        )

                    seqlens_ = seqlens_[n_seqs:]
                    cu_seqlens_ = (
                        cu_seqlens_[n_seqs:] - np.sum(cu_seqlens_[n_seqs - 1]).item()
                    )
                expanded_seqs = []
                for i in range(batch_size):
                    expanded_seqs.extend(
                        np.split(
                            seqs[i], indices_or_sections=np.cumsum(seqlens[i][:-1])
                        )
                    )
                seqs = expanded_seqs

                expanded_states = []
                for i in range(batch_size):
                    expanded_states.extend(
                        np.split(
                            states[i], indices_or_sections=np.cumsum(seqlens[i][:-1])
                        )
                    )
                states = expanded_states

                seqlens = torch.Tensor([len(seq) for seq in seqs])

                # Simply replace a random suffix length with padding
                ignore_mask = (
                    torch.arange(seqlens.max()).tile(len(seqs), 1) >= seqlens[:, None]
                )
                seqs = pad_sequence(
                    seqs,
                    batch_first=True,
                )
                states = pad_sequence(
                    states,
                    batch_first=True,
                )
                out_dict.update({"input_ids": seqs, "states": states})
            else:
                seqlens = torch.Tensor(
                    self.generator.integers(
                        low=length[0], high=length[1] + 1, size=batch_size
                    )
                )

                ignore_mask = (
                    torch.arange(length[1]).tile(len(seqs), 1) >= seqlens[:, None]
                )

                out_dict.update({"ignore_mask": ignore_mask})

        if self.cfg.start_at_n != None:
            assert isinstance(self.cfg.start_at_n, int)
            assert variable_len == False, "Cannot use <start_at_n> with variable length"
            ignore_mask = torch.arange(length[1]).tile(len(seqs), 1) < self.cfg.start_at_n
            out_dict.update({"ignore_mask": ignore_mask})

        return out_dict


class SubsetIntervened(Dataset):
    r"""
    Dataset of sequences where the underlying HMM undergoes an intervention during generation

    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    dataset: CompositionalHMMDataset
    prefix_indices: Sequence[int]

    def __init__(
        self,
        dataset: CompositionalHMMDataset,
        prefix_indices: Sequence[int],
        suffix_indices: Sequence[int],
        intv_idx: Union[Tuple[int], int],
    ) -> None:
        self.dataset = dataset
        self.prefix_indices = prefix_indices
        self.suffix_indices = suffix_indices
        self.intv_idx = intv_idx

    # Only support batched getitems like in the HMM dataset (for simplicity and efficiency)
    def __getitems__(self, indices: List[int]):
        return self.dataset.__getitems__(
            envs=[self.prefix_indices[idx] for idx in indices],
            intv_envs=[self.suffix_indices[idx] for idx in indices],
            intv_idx=self.intv_idx,
        )

    def __len__(self):
        return len(self.prefix_indices)


class PrecomputedDataset(Dataset):
    def __init__(self, data_dict):
        """
        Args:
            data_dict (dict): A dictionary where each value is a tensor of the same length.
        """
        self.data_dict = data_dict
        self.keys = list(data_dict.keys())
        self.length = data_dict[self.keys[0]].size(0)  # Length of the dataset

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Return a dictionary with the same structure as self.data_dict, indexed by idx
        return {key: tensor[idx] for key, tensor in self.data_dict.items()}

    def __getitems__(self, indices):
        # Return a dictionary with the same structure as self.data_dict, indexed by idx
        return {key: tensor[indices] for key, tensor in self.data_dict.items()}
