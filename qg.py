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
import qgpy
import qgpy.submit
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from qgpy.configuration import QGConfig
import os
import logging
import dataclasses


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

    print(f"Pythia configuration (unresolved): {cfg}")
    print(type(cfg))
    p = dataclasses.asdict(cfg)
    logger.info(f"Pythia configuration: {p}")

    # Get the output directory from the Hydra runtime configuration.
    # This is the directory where all the outputs will be aggregated in the end.
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    # run_dir is a directory to which each job will write its outputs when running.
    # Each job will create a subdirectory in the run_dir with its own name.
    # If cfg.general.run_dir is an empty string, set run_dir to the output_dir.
    # If it is not empty, set run_dir to the value of cfg.general.run_dir.
    # In this case, it will be copied to the output_dir at the end of the run.
    if cfg.general.run_dir:
        run_dir = cfg.general.run_dir
    else:
        run_dir = output_dir

    # For each job, prepare the job dictionary and store it in a list.
    jobs = []

    for i, (slice_min, slice_max, njobs) in enumerate(zip(cfg.slicing.slices_min, cfg.slicing.slices_max, cfg.slicing.njobs)):
        for j in range(njobs):

            job_name = f"slice{i}_{j}"
            job = {
                'job_name': job_name,
                'job_dir': f"{run_dir}/{job_name}",
                'target_dir': output_dir,
                'cfg': getattr(cfg, cfg.general.generator),
                'delphes_card': cfg.general.delphes_card,
                'slice_min': slice_min,
                'slice_max': slice_max,
            }
            jobs.append(job)

    # Submit the jobs.
    if jobs:
        logger.info(f"Prepared {len(jobs)} jobs for submission. Submitting them now...")
        # The submission function must have the same name as the scheduler specified in the configuration.
        f = getattr(qgpy.submit, cfg.submit.scheduler)
        f(cfg = getattr(cfg.submit, cfg.submit.scheduler), jobs = jobs)

    # Print a message indicating that the submission part is over.
    logger.info("Submission of jobs is over.")
    logger.info("-----------------------------------------------------------")






    # Print a message indicating that the script finished successfully.
    logger.info("qg.py script finished successfully.")
    logger.info("-----------------------------------------------------------")

# Call the main function.
if __name__ == "__main__":
    main()
