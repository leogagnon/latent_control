# Setup
- Create a venv : `python3.10 -m venv venv`
- Add the following lines to `venv/bin/activate` : 
```
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
module load libreadline/7.0  OpenSSL/1.1 python/3.10 libffi/3.2.1 cuda/12.1.1/cudnn/9.1 cudatoolkit/12.1.1
```
- Activate env : `source venv/bin/activate`
- Install requirements (**ignoring conflicts**) : `pip install --no-dependencies -r requirements.txt`
- Make
# Example usage
- Run on current session : `python train_metalearn.py task/model=gpt5M task.val_size=420`
- Run sweep : `python train_metalearn.py -m hydra/sweeper/params=basic`

# Notes
- When using checkpointing, the files are saved **localy** at the designed location and under a folder named by its WANDB run ID. Checkpointing is disabled by default to prevent saving useless stuff.
  - Then you can load a run simply as `task = MetaLearningTask(run_id)`
