"""
The main script of the qgpy package.

Usage
-----
To use this module, ensure Hydra is installed:
    
    pip install hydra-core
    
Run a script using Hydra with CLI overrides:

    python qg.py config.param=value

Note that you can profit from the tab completion feature of Hydra.
To enable it, run the command suggested in the output of the following command:

$ python qg.py --hydra-help

Examples
--------
Override the number of jobs to be submitted:

$ python qg.py njobs=100

Change the config directory path and the config file name:

$ python qg.py --config-name=config_user --config-path=config_user

Change the hydra output directory:

$ python qg.py hydra.run.dir=outputs/2025-08-08/09-31-56
"""

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from qgpy.configuration import QGConfig
import os
import logging

# The singleton instance of the ConfigStore is necessary
# for the structured configuration management.
cs = ConfigStore.instance()
cs.store(name="base_config", node=QGConfig)

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg) -> None:
    """
    The main function of the script.

    Parameters
    ----------
    cfg : DictConfig
        The configuration object created by Hydra.
    """

    # Get the logger.
    logger = logging.getLogger("qgpy")
    # logger.setLevel(cfg.log.qgpy)
    # logger.propagate = False

    # Print the configuration.
    logger.info(OmegaConf.to_yaml(cfg))

    # Convert DictConfig to an object where nested attributes
    # become instances of your dataclasses.
    cfg = OmegaConf.to_object(cfg)

    # Get the output directory from the Hydra runtime configuration.
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    # Print a message indicating that the script finished successfully.
    logger.info("qg.py script finished successfully.")
    logger.info("-----------------------------------------------------------")

# Call the main function.
if __name__ == "__main__":
    main()
