from data.hmm import CompositionalHMMDataset, CompositionalHMMDatasetConfig
import jax.numpy as jnp
import jax.random as jr
from data.active import HMMEnv
from torch2jax import t2j


def test_data():
    hmm = CompositionalHMMDataset(
        CompositionalHMMDatasetConfig(seed=0, has_actions=True)
    )

    hmm_ = CompositionalHMMDataset(
        CompositionalHMMDatasetConfig(seed=0, has_actions=False)
    )
    assert jnp.all(hmm.latent_transmat == hmm_.latent_transmat)
    assert jnp.all(hmm_.latent_emissions == hmm_.latent_emissions)

    X, Z = hmm.sample(1337, 200, jr.PRNGKey(0))
    oracle_dict = hmm.bayesian_oracle(jnp.arange(len(hmm)), X)
    hmm_post = jnp.exp(oracle_dict["log_alpha_post"][100])
    assert hmm_post.max() == 1.0
    assert hmm_post.argmax() == 1337
    assert oracle_dict["z_post"][100].argmax() == Z[99]
    oracle_dict_ = hmm.bayesian_oracle(
        jnp.arange(len(hmm)),
        X[100:],
        initial_messages=oracle_dict["messages"][99],
        log_alpha_prior=oracle_dict["log_alpha_post"][100],
    )
    assert jnp.allclose(
        oracle_dict["messages"][100:], oracle_dict_["messages"], atol=1e4
    )


def test_env():
    hmm = CompositionalHMMDataset(
        CompositionalHMMDatasetConfig(seed=0, has_actions=True)
    )
    hmm_env = HMMEnv(hmm, 0)
    hmm_env.reset(1)
    for i in range(199):
        hmm_env.step(jnp.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]))
    X = t2j(hmm_env.history[0, 0::2])
    hmm_post = jnp.exp(hmm.bayesian_oracle(jnp.arange(len(hmm)), X)["log_alpha_post"])[
        100
    ]
    assert hmm_post.max() == 1.0
    assert hmm_post.argmax() == hmm_env.indices[0].item()
