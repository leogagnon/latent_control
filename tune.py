from train import *

if __name__ == "__main__":
    hydra_wrapper = hydra.main(version_base=None, config_name="tune", config_path="configs/")
    hydra_wrapper(main)()
