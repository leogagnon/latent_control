# Setup
- Create a venv : `python3.10 -m venv venv`
- Add the following lines to `venv/bin/activate` : 
```
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
module load libreadline/7.0  OpenSSL/1.1 python/3.10 libffi/3.2.1 cuda/12.1.1/cudnn/9.1 cudatoolkit/12.1.1
```
- Install requirements (ignoring conflicts because this is a mess) : `pip --no-deps -r requirements.txt`
- Wait
# Example usage
- Run on current session : `python train_metalearn.py task/model=gpt5M task.val_size=420`
- Run sweep : `python train_metalearn.py -m hydra/sweeper/params=basic`