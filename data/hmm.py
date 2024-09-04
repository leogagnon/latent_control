from math import gcd
from hmmlearn import hmm
import numpy as np
from scipy.special import logsumexp, softmax
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


@dataclass
class CompositionalHMMDatasetConfig:
    n_states: int = 30
    n_obs: int = 60
    context_length: Tuple[int] = (6,6)
    context_length_dist: Optional[str] = None
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


def cycle_to_transmat(cycle: List[int], n_states: int) -> np.array:

    transition = np.zeros(shape=(n_states, n_states), dtype=np.int16)
    for i in range(len(cycle)):
        transition[
            cycle[i],
            cycle[(i + 1) % (len(cycle))],
        ] = 1.0
    return transition


def preferential_attachement_edges(
    states: np.array, obs: np.array, n: int
) -> List[Tuple[int]]:

    states_ids = np.arange(len(states))
    obs_ids = np.arange(len(obs))
    obs_degree = np.zeros_like(obs_ids)

    # Add initial edges from each state to a random obs
    init_obs = np.random.choice(obs_ids, size=len(states), replace=False)
    obs_degree[init_obs] = 1
    edges = [(states[i], obs[init_obs[i]]) for i in range(len(states))]

    # Iteratively add <n> edges, sampling state at random and obs at random weighted by degree
    for i in range(n):
        s_id = np.random.choice(states_ids)
        o_id = np.random.choice(obs_ids, p=softmax(obs_degree))

        obs_degree[o_id] += 1
        edges.append((states[s_id], obs[o_id]))

    return edges


class CompositionalHMMDataset(Dataset):
    def __init__(self, cfg: CompositionalHMMDatasetConfig) -> None:
        super().__init__()

        print("Initializing dataset...", end="")
        self.cfg = cfg
        self.index_to_latent = self._make_index_to_latent()
        self.latent_transmat = self._make_env_transition()
        self.latent_emissions = self._make_env_emission()
        print("Done!")

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
            base_cycle = np.random.permutation(np.arange(len(states)))
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
                    np.random.choice(states, size=length, replace=False)
                    for length in np.random.randint(3, 9, size=self.cfg.cycle_per_group)
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
                    group, obs, self.cfg.emission_edge_per_node * len(group)
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

    def get_hmm(self, index: int):

        latent = self.index_to_latent[index]
        transition_latent = latent[: (3 + self.cfg.cycle_families + 2)]
        emission_latent = latent[-(self.cfg.emission_groups + 1) :]

        model = hmm.CategoricalHMM(
            n_components=self.cfg.n_states, n_features=self.cfg.n_obs
        )

        model.transmat_ = self.latent_transmat[tuple(transition_latent)]
        model.emissionprob_ = self.latent_emissions[tuple(emission_latent)]
        model.startprob_ = np.array(
            [1 / self.cfg.n_states] * self.cfg.n_states
        )  # uniform

        return model

    def log_fwd(self, X, num_workers: int = 1, latent_indices=None) -> np.array:
        """
        Computes the forward algorithm on X for all latents specified by <latent_indices>.
        Potentially operates in parallel since this is slow.

        P(x_{1...t}, z_t | latent)

        Args:
            X : Inputs
            num_workers (int, optional): Number of parallel workers
            latent_indices : What latents to perform the algorithm on. Defaults to None.

        Returns:
            np.array : Array of shape [latent_indices, len(X), n_states]
        """

        # Define woker
        def log_fwd_worker(data: CompositionalHMMDataset, X, indices, conn: Connection):

            log_fwds = np.zeros(
                shape=(len(indices), len(X), data.cfg.n_states), dtype=np.float16
            )
            for i, idx in enumerate(indices):
                model = data.get_hmm(idx)
                log_fwds[i] = _hmmc.forward_log(
                    model.startprob_, model.transmat_, model._compute_log_likelihood(X)
                )[1]
            conn.send(log_fwds)

        if latent_indices is None:
            latent_indices = np.arange(len(self))
        splits = np.array_split(latent_indices, num_workers)
        processes = []
        conns = []
        for i in range(num_workers):
            c1, c2 = mp.Pipe()
            conns.append(c1)
            process = mp.Process(target=log_fwd_worker, args=(self, X, splits[i], c2))
            processes.append(process)
            process.start()

        log_fwd = np.zeros(
            (len(latent_indices), len(X), self.cfg.n_states), dtype=np.float16
        )
        for i in range(len(splits)):
            log_fwd[splits[i]] = conns[i].recv()

        for process in processes:
            process.join()

        return log_fwd

    def log_likelihood(self, X, constraint=None, num_workers: int = 1):
        """p(X | constraints)"""

        if constraint is None:
            latent_indices = None
        else:
            latent_indices = np.nonzero(
                self.index_to_latent[:, constraint[0]] == constraint[1]
            )[0]
        fwd = self.log_fwd(X, num_workers=num_workers, latent_indices=latent_indices)

        # log-likelihood for every latent
        lls = logsumexp(fwd[:, -1], axis=-1)
        # integrate of all latents (divide by len(lls))
        ll = logsumexp(lls) - np.log(len(lls))

        return ll

    def posterior_predictive(self, X, num_workers: int = 1):
        """p(x_{t+1} | x_{1...t}) for all t"""

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # log_p(z_{t}, x_{1..t} | alpha)
        log_fwd = self.log_fwd(X, num_workers=num_workers)
        log_fwd = torch.from_numpy(log_fwd).to(dtype=torch.float16, device=device)

        # log_p(x_{1..t} | alpha) = sum_{z_t} p(z_t, x_{1...t} | alpha)
        log_like = torch.logsumexp(log_fwd, dim=-1)

        # p(z_t | x_{1...t}, alpha) = p(z_t, x_{1...t} | alpha) / p(x_{1...t} | alpha)
        z_post = torch.nan_to_num(torch.exp(log_fwd - log_like[..., None]), nan=0)

        latent_transmat_shape = self.latent_transmat.shape[:-2]
        latent_emissions_shape = self.latent_emissions.shape[:-2]
        latent_shape = latent_transmat_shape + latent_emissions_shape

        # p(z_{t+1} | z_t, alpha)
        trans = (
            torch.broadcast_to(
                torch.from_numpy(
                    self.latent_transmat[
                        (...,)
                        + (None,) * len(latent_emissions_shape)
                        + (slice(None),) * 2
                    ]
                ),
                latent_shape + (self.cfg.n_states, self.cfg.n_states),
            )
            .reshape(-1, self.cfg.n_states, self.cfg.n_states)
            .to(device=device)
        )

        # p(x_{t+1} | z_{t+1}, alpha)
        emission = (
            torch.broadcast_to(
                torch.from_numpy(
                    self.latent_emissions[
                        (None,) * len(latent_transmat_shape)
                        + (...,)
                        + (slice(None),) * 2
                    ]
                ),
                latent_shape + (self.cfg.n_states, self.cfg.n_obs),
            )
            .reshape(-1, self.cfg.n_states, self.cfg.n_obs)
            .to(device=device)
        )

        # p(x_{t+1} | x_{1...t}, alpha) = sum_z p(x_{t+1} | z_{t+1}, alpha) p(z_{t+1} | z_t, alpha) p(z_t | x_{1...t}, alpha)
        log_pp_given_alpha = torch.log(
            torch.einsum("atz,azv,avx->atx", z_post, trans, emission)
        )

        # p(alpha | x_{1...t}) = p(x_{1...t} | alpha) p(alpha) / sum_{alpha} p(x_{<t} | alpha) p(alpha)
        log_alpha_post = log_like - math.log(len(self))
        log_alpha_post = log_alpha_post - torch.logsumexp(log_alpha_post, dim=0)[None]

        # p(x_t | x_{<t}) = \sum_{alpha} p(x_t | x_{<t}, alpha) p(alpha | x_{<t})
        pp = torch.nan_to_num(
            torch.exp(
                torch.logsumexp(log_pp_given_alpha + log_alpha_post[..., None], dim=0)
            ),
            nan=0.0,
        )

        return pp.cpu().numpy()

    def latent_posterior(self, X, id: int, num_workers: int = 1):
        """p(latent_id | X)"""

        device = "cuda"

        fwd = self.log_fwd(X, num_workers=num_workers)
        fwd = torch.from_numpy(fwd).to(device=device)

        lls = torch.logsumexp(fwd[:, -1], dim=-1)
        index_to_latent = torch.from_numpy(self.index_to_latent).to(device=device)
        len_latent = len(torch.unique(index_to_latent[:, id]))
        env_per_latent = len(lls) // len_latent

        latent_lls = []
        for j in range(len_latent):
            mask = index_to_latent[:, id] == j
            latent_lls.append(
                torch.logsumexp(lls[mask], dim=0) - math.log(env_per_latent)
            )
        latent_lls = torch.Tensor(latent_lls)

        score = torch.exp(latent_lls - logsumexp(latent_lls))

        return score.cpu().numpy()

    def __getitem__(
        self, index: int, n_step: Optional[int] = None, seed: Optional[int] = None
    ):
        if n_step is None:
            if isinstance(self.cfg.context_length, int):
                n_step = self.cfg.context_length
            else:
                if self.cfg.context_length_dist == "uniform":
                    n_step = np.random.randint(
                        self.cfg.context_length[0], self.cfg.context_length[1]
                    )
                elif self.cfg.context_length_dist == "exponential":
                    r = self.cfg.context_length[1] - self.cfg.context_length[0]
                    n_step = (
                        r
                        * np.clip(np.random.exponential(scale=1.5), a_min=0, a_max=5.0)
                        / 5
                    ) + self.cfg.context_length[0]

        model = self.get_hmm(index)
        X = model.sample(int(n_step), random_state=seed)[0].squeeze()
        

        return np.concatenate([X, -np.ones(self.cfg.context_length[1] - len(X), dtype=np.int32)])
