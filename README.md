# ICML2025 Workshop (ES-FOMO III)
- Paper : [Next-Token Prediction Should be Ambiguity-Sensitive: A Meta-Learning Perspective](https://openreview.net/forum?id=vE0bhgmDze)
- Sweeps in `configs/hydra/sweeper/params/workshop`
- Figure generation in `notebooks/workshop_figures.ipynb`

# Setup
1) Install python 3.10
2) Create a venv : `python3.10 -m venv venv`
3) Add the following lines to `venv/bin/activate` : 
```
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
module load python/3.10 cuda/12.1.1/cudnn/9.1 cudatoolkit/12.1.1
```
4) Activate env : `source venv/bin/activate`
5) Install requirements : `pip install -r requirements.txt`

# Example usage
- Run on current session : `python train.py task/metalearn=base task/metalearn/data=small task.metalearn.lr=0.01`
  
- Run on new session (with submitit) : 
  - Sweep : `python train.py -m hydra/sweeper/params=workshop/diffusion`
  - Single run : `python train.py -m task/metalearn=...`
