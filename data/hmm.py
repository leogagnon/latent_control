from hmmlearn import hmm
import numpy as np
from scipy.special import logsumexp
from torch.utils.data import Dataset, Subset
from itertools import product
import lightning as L
from typing import *
from dataclasses import dataclass


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
        with np.errstate(divide='ignore', invalid='ignore'):
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

            nodes.update(
                rng.choice(list(free_nodes), subgraph_size, replace=False)
            )
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
    
    def likelihood(self, X, constraints):
        good_latents = self.index_to_latent[self.index_to_latent[:,constraints[0]] == constraints[1]]
        Zs = np.zeros(len(good_latents))
        model = self.get_hmm(0)
        for i in range(len(good_latents)):
            model.transmat_ = self.get_transmat(good_latents[i])
            Zs[i] = model._score_log(X, lengths=None, compute_posteriors=False)[0]
        return logsumexp(Zs)
    
    def lc_score(self, X, latent: int):
        l0 = self.likelihood(X, (latent,False))
        l1 = self.likelihood(X, (latent,True))
        return np.e**l0/(np.e**logsumexp([l0,l1]))

    def __getitem__(self, index: int, n_step: Optional[int] = None, seed: int = None):
        if n_step is None:
            n_step = self.cfg.context_length

        model = self.get_hmm(index)
    
        return model.sample(n_step)[0].squeeze()
    

