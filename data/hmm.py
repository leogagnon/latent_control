from hmmlearn import hmm
import numpy as np
from torch.utils.data import Dataset
from itertools import product
import lightning as L
from typing import *
from dataclasses import dataclass


@dataclass
class CompositionalHMMDatasetConfig:
    n_latents: int = 16
    n_states: int = 32
    n_overlap: int = 2
    min_active_latents: int = 4
    max_active_latents: int = 8
    context_length: int = 6


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
        subgraph_size = self.cfg.n_states // self.cfg.n_latents
        free_nodes = set(range(1, self.vocab_size))
        used_nodes = set([])

        subgraphs_nodes = []
        for _ in range(self.cfg.n_latents):
            nodes = set([])

            nodes.update(
                np.random.choice(list(free_nodes), subgraph_size, replace=False)
            )
            if len(used_nodes) != 0:
                nodes.update(
                    np.random.choice(
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

    def __getitem__(self, index: int, n_step: Optional[int] = None):
        if n_step is None:
            n_step = self.cfg.context_length
        model = hmm.CategoricalHMM(
            n_components=self.vocab_size, n_features=self.vocab_size
        )

        model.transmat_ = self.get_transmat(self.index_to_latent[index])
        model.startprob_ = np.array([1.0] + ([0.0] * self.cfg.n_states))
        model.emissionprob_ = self.emission

        return model.sample(n_step)[0].squeeze()
