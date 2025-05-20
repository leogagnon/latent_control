# Setup
- Load python3.10 : `module load python3.10`
- Create a venv : `python3.10 -m venv venv`
- Activate env : `source venv/bin/activate`
- Install requirements (lol) : `pip install --no-dependencies -r requirements.txt`
# Example usage
- Run on current session :
  - `python train.py task/metalearn=...`
  - `python train.py task/diffusion=... task.diffusion.model.n_embd=512`
  
- Run on new session (with submitit) : 
  - Sweep : `python train.py -m hydra/sweeper/params=metalearn/base`
  - Single run : `python train.py -m task/metalearn=...`
