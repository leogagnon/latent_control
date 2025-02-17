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
- Run on current session :
  - `python train.py task/metalearn=...`
  - `python train.py task/gfn=... task.gfn.train_direction=fwd`
  
- Run on new session (with submitit) : 
  - Sweep : `python train.py -m hydra/sweeper/params=metalearn/base`
  - Single run : `python train.py -m task/metalearn=...`