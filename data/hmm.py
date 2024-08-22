from hmmlearn import hmm
import numpy as np
from scipy.special import logsumexp
import torch
from torch.utils.data import Dataset, Subset
from itertools import product
import lightning as L
from typing import *
from dataclasses import dataclass
from hmmlearn.base import _hmmc
from tqdm import tqdm
import multiprocessing as mp


@dataclass
class CompositionalHMMDatasetConfig:
    n_latents: int = 10
    n_states: int = 20
    n_overlap: int = 1
    min_active_latents: int = 3
    max_active_latents: int = 7
    context_length: int = 6
    seed: int = 42


class CompositionalHMMDataset(Dataset):
    def __init__(self, cfg: CompositionalHMMDatasetConfig) -> None:
        super().__init__()

        print("Initializing dataset...", end="")
        self.cfg = cfg
        self.vocab_size = cfg.n_states + 1
        self.index_to_latent = self._make_index_to_latent()
        self.latent_transmat = self._generate_cycle_composition_graph()
        self.emission = np.eye(self.vocab_size, self.vocab_size)
        print("Done!")

    def get_transmat(self, latents: np.array):
        with np.errstate(divide="ignore", invalid="ignore"):
            transmat = self.latent_transmat[latents].sum(0)
            transmat = transmat / transmat.sum(1)[:, None]
            transmat = np.nan_to_num(transmat, nan=0.0)
            transmat[transmat.sum(0) == 0, 0] = 1.0

        return transmat

    def _make_index_to_latent(self):
        n_latents = self.cfg.n_latents
        max_active_latents = self.cfg.max_active_latents
        min_active_latents = self.cfg.min_active_latents

        all_binary_strings = ["".join(bits) for bits in product("01", repeat=n_latents)]
        filtered_strings = [
            list(s)
            for s in all_binary_strings
            if min_active_latents < s.count("1") < max_active_latents
        ]

        return np.array(filtered_strings, dtype=int).astype(bool)

    def _generate_cycle_composition_graph(self):

        assert self.cfg.n_states % self.cfg.n_latents == 0
        rng = np.random.RandomState(self.cfg.seed)

        subgraph_size = self.cfg.n_states // self.cfg.n_latents
        free_nodes = set(range(1, self.vocab_size))
        used_nodes = set([])

        subgraphs_nodes = []
        for _ in range(self.cfg.n_latents):
            nodes = set([])

            nodes.update(rng.choice(list(free_nodes), subgraph_size, replace=False))
            if len(used_nodes) != 0:
                nodes.update(
                    rng.choice(
                        list(used_nodes),
                        min(self.cfg.n_overlap, len(used_nodes)),
                        replace=False,
                    )
                )
            free_nodes.difference_update(nodes)
            used_nodes.update(nodes)

            subgraphs_nodes += [[0] + list(nodes)]

        subgraphs_mat = np.zeros(
            shape=(self.cfg.n_latents, self.vocab_size, self.vocab_size)
        )
        for i in range(self.cfg.n_latents):
            for j in range(len(subgraphs_nodes[i])):
                subgraphs_mat[
                    i,
                    subgraphs_nodes[i][j],
                    subgraphs_nodes[i][(j + 1) % (len(subgraphs_nodes[i]))],
                ] = 1.0

        return subgraphs_mat

    def __len__(self):
        return len(self.index_to_latent)

    def get_hmm(self, index: int):

        model = hmm.CategoricalHMM(
            n_components=self.vocab_size,
            n_features=self.vocab_size,
        )
        model.transmat_ = self.get_transmat(self.index_to_latent[index])
        model.startprob_ = np.array([1.0] + ([0.0] * self.cfg.n_states))
        model.emissionprob_ = self.emission

        return model

    def log_fwd(self, X, num_workers: int = 1, latent_indices=None) -> np.array:
        """
        Computes the forward algorithm on X for all latents specified by <latent_indices>.
        Potentially operates in parallel since this is slow

        Args:
            X : Inputs
            num_workers (int, optional): Number of parallel workers
            latent_indices : What latents to perform the algorithm on. Defaults to None.

        Returns:
            np.array : Array of shape [latent_indices, len(X), n_latents]
        """

        # Define woker
        def log_fwd_worker(data: CompositionalHMMDataset, X, indices, conn):

            fwds = np.zeros(shape=(len(indices), len(X), data.cfg.n_states + 1))
            model = data.get_hmm(0)
            for i, latent in enumerate(data.index_to_latent[indices]):
                model.transmat_ = data.get_transmat(latent)
                fwds[i] = _hmmc.forward_log(
                    model.startprob_, model.transmat_, model._compute_log_likelihood(X)
                )[1]

            conn.send(fwds)

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

        arrs = [c.recv() for c in conns]
        [process.join() for process in processes]

        log_fwd = np.concatenate(arrs)

        return log_fwd

    def log_likelihood(self, X, constraints=None):

        if constraints is None:
            latent_indices = None
        else:
            latent_indices = np.nonzero(
                self.index_to_latent[:, constraints[0]] == constraints[1]
            )[0]
        fwd = self.log_fwd(X, num_workers=4, latent_indices=latent_indices)

        # log-likelihood for every latent
        lls = logsumexp(fwd[:, -1], axis=-1)
        # integrate of all latents (divide by len(lls))
        ll = logsumexp(lls) - np.log(len(lls))

        return ll

    def posterior_predictive(self, X):
        
        log_fwd = self.log_fwd(X, num_workers=4)

        log_like = logsumexp(log_fwd, axis=-1) # log_p(x_{<t} | theta)
        post = np.nan_to_num(np.exp(log_fwd - log_like[..., None]), nan=0.0) # p(z_{t-1} | x_{<t}, \theta)
        
        dummy = self.get_hmm(0)
        trans = np.zeros((len(self), self.vocab_size, self.vocab_size))  # p(z_t | z_{t-1}, \theta)
        emission = np.zeros((len(self), self.vocab_size, self.vocab_size))  # p(x_t | z_t, \theta)
        for i in range(len(self)):
            trans[i] = self.get_transmat(self.index_to_latent[i])
            emission[i] = dummy.emissionprob_

        pp_given_theta = np.einsum(
            "atz,azv,avx->atx", post, trans, emission, optimize=True
        )
        pp = logsumexp(np.log(pp_given_theta) + log_like[..., None], axis=0)
        pp = pp - logsumexp(pp, axis=-1, keepdims=True)
        pp = np.exp(pp)
        pp = np.nan_to_num(pp, nan=0.0)

        return pp

    def lc_score(self, X, latent: int):

        fwd = self.log_fwd(X, num_workers=4)

        lls = logsumexp(fwd[:, -1], axis=-1)

        latent_mask = self.index_to_latent[:, latent] == False

        ll0 = logsumexp(lls[latent_mask]) - np.log(latent_mask.sum())
        ll1 = logsumexp(lls[~latent_mask]) - np.log((~latent_mask).sum())

        score = np.exp(ll0 - (logsumexp([ll0, ll1])))

        return score

    def __getitem__(self, index: int, n_step: Optional[int] = None, seed: int = None):
        if n_step is None:
            n_step = self.cfg.context_length

        model = self.get_hmm(index)

        return model.sample(n_step)[0].squeeze()
