import warnings

warnings.filterwarnings("ignore", message="invalid value encountered in divide")
warnings.filterwarnings("ignore", message="Trying to infer the `batch_size` from an ambiguous collection.")
warnings.filterwarnings("ignore", message="The `srun` command is available on your system but is not used. ")
warnings.filterwarnings("ignore", message="Because the driver is older than the PTX compiler version, XLA is disabling parallel compilation, which may slow down compilation.")
warnings.filterwarnings("ignore", message="The number of training batches ")
warnings.simplefilter(action='ignore', category=FutureWarning)

import hydra

from train_metalearn import main

# Just wraps the main() function of train.py with a different base config : tune.yaml
if __name__ == "__main__":
    hydra_wrapper = hydra.main(version_base=None, config_name="tune", config_path="configs_metalearn/")
    hydra_wrapper(main)()
