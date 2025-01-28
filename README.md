# Setup
- Load python3.10 : `module load python3.10`
- Create a venv : `python3.10 -m venv venv`
- Add the following lines to `venv/bin/activate` : 
```
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
module load python/3.10 cuda/12.1.1/cudnn/9.1 cudatoolkit/12.1.1
```
- Activate env : `source venv/bin/activate`
- Install requirements (lol) :
  - `pip install --no-dependencies -r requirements.txt`
  - `pip install wheel`
  - `pip install --no-deps --no-cache-dir causal-conv1d==1.4.0`
  - `pip install --no-deps --no-cache-dir --no-build-isolation mamba-ssm==2.2.4`
# Example usage
- Run on current session : `python train_metalearn.py task/model=gpt5M task.val_size=420`
- Run sweep (with submitit) : `python train_metalearn.py -m hydra/sweeper/params=basic`

# Notes
- Diffusion implementation inspired from Latent Diffusion for Language Generation ([paper](https://arxiv.org/abs/2212.09462),[code](https://github.com/justinlovelace/latent-diffusion-for-language))
- When using checkpointing, the files are saved **localy** at the designed location and under a folder named by its WANDB run ID. Checkpointing is disabled by default to prevent saving useless stuff.
  - Then you can load a run simply as `task = MetaLearningTask(run_id)`
