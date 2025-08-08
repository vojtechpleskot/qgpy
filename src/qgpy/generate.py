"""
generate.py
===========

Functions for generating MC events with different MC generators and/or their settings.

Functions
---------
generate_pythia(Dict[str, Any]) -> None
    Generate events using the Pythia event generator with the specified configuration.
"""

import os
from typing import Dict, Any
from qgpy import utils

def generate_pythia(outdir: str, cfg: Dict[str, Any]) -> None:
    """
    Generate events using the Pythia event generator with the specified configuration.

    Parameters
    ----------
    outdir : str
        The output directory where the generated events will be stored.
    cfg : Dict[str, Any]
        The configuration dictionary containing the Pythia settings.
    """

    # Ensure the output directory exists.
    os.makedirs(outdir, exist_ok=True)
    
    # Create a logger instance.
    logger = utils.create_logger('generate', outdir = outdir, level = cfg['log_level'])
    logger.info("Starting the event generation...")

    # Prepare the command to run Pythia.
    command = f"{cfg['executable']} --nevents={cfg['nevents_per_job']} --seed={cfg['seed']} " \
                f"--pTHatMin={cfg['pTHatMin']} --pTHatMax={cfg['pTHatMax']} " \
                f"--output={outdir}/generate --jetPtMin={cfg['reco_jet_pt_min']}"

    # Execute the command in the shell.
    logger.info(f"Running command: {command}")
    exit_code = os.system(command)

    # Throw an error if the command failed.
    if exit_code != 0:
        logger.error(f"Event generation failed with exit code {exit_code}.")
        raise RuntimeError(f"Event generation failed with exit code {exit_code}.")
    
    # Return.
    return