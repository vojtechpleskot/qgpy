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






    # Jobs will contain the configuration of each generation job.
    jobs = []

    # Use Cartesian product to generate all combinations.
    # Loop over all combinations to perform the fit and calculate the UFFs.
    for combination in itertools.product(*bin_lists):

        # Create a dictionary with the variable:interval structure.
        # This dictionary is a representation of the phase space bin.
        bin = {v: interval for v, interval in zip(variables_order, combination)}

        # Create the output directory for the fit results.
        fit_out_dir = f"{output_dir}/{cfg_uff.fit.outdir}/{histogram_name('bin', variables_order, bin)}"
        logger.info(f"Creating output directory: {fit_out_dir}")
        os.makedirs(fit_out_dir, exist_ok = True)

        # Perform the fit.
        # If fit_result.pkl already exists, then skip this step.
        if os.path.exists(f"{fit_out_dir}/fit_result.pkl"):
            logger.info(f"Fit result already exists for bin {bin}, and will be loaded from file.")
        else:
            job_config = {
                'hists'                    : hists,
                'templates'                : cfg_uff.fit.templates,
                'sr'                       : cfg_uff.fit.sr,
                'wp'                       : cfg_uff.ff.wp,
                'trig'                     : cfg_uff.ff.trig,
                'data_name'                : cfg_uff.ff.data_name,
                'real_name'                : cfg_uff.ff.real_name,
                'bin'                      : bin,
                'variables_order'          : variables_order,
                'variables_info'           : cfg_uff.ff.variables_info,
                'plot_opts'                : cfg_uff.plot_data_vs_model,
                'fit_out_dir'              : fit_out_dir,
                'backend'                  : cfg_uff.fit.backend,
                'nf_bounds'                : cfg_uff.fit.nf_bounds,
                'disc_var_label'           : cfg_uff.disc_var.label,
                'fit_result_file_name'     : f"{fit_out_dir}/fit_result.pkl",
                'post_fit_hists_file_name' : f"{fit_out_dir}/post_fit_hists.pkl",
            }
            jobs.append(job_config)

    # Submit the fit jobs.
    if jobs:
        logger.info(f"Prepared {len(jobs)} fit jobs for submission. Submitting them now...")
        # The submission function must have the same name as the scheduler specified in the configuration.
        f = getattr(ufftools.submit, cfg_uff.submit.scheduler)
        f(cfg = getattr(cfg_uff.submit, cfg_uff.submit.scheduler), jobs = jobs)

    # Print a message indicating that the fitting part is over.
    logger.info("Fitting is over.")
    logger.info("-----------------------------------------------------------")






    # Print a message indicating that the script finished successfully.
    logger.info("qg.py script finished successfully.")
    logger.info("-----------------------------------------------------------")

# Call the main function.
if __name__ == "__main__":
    main()
