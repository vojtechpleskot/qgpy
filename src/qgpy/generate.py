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
import qgpy
import qgpy.utils
from qgpy.configuration import GeneratorConfig

def generate_pythia(
        outdir: str,
        cfg: GeneratorConfig,
        slice_min: float = -1,
        slice_max: float = -1,
        ) -> None:
    """
    Generate events using the Pythia event generator with the specified configuration.

    Parameters
    ----------
    outdir : str
        The output directory where the generated events will be stored.
    cfg : GeneratorConfig
        The configuration object containing the MC generator settings.
    slice_min : float, optional
        The minimum value of the slicing variable pT hat.
        When -1, the Pythia 8 default is used.
    slice_max : float, optional
        The maximum value of the slicing variable pT hat.
        When -1, the Pythia 8 default is used.
    """

    # Ensure the output directory exists.
    os.makedirs(outdir, exist_ok=True)
    
    # Create a logger instance.
    logger = qgpy.utils.create_logger('generate', outdir = outdir, level = cfg.log_level)
    logger.info("Starting the event generation...")

    # Prepare the command to run Pythia.
    package_dir = os.path.dirname(os.path.abspath(qgpy.__file__))
    command = f"{package_dir}/../../{cfg.executable} --nevents={cfg.nevents_per_job} --seed={cfg.seed} " \
                f"--pTHatMin={slice_min} --pTHatMax={slice_max} " \
                f"--output={outdir}/generate --recoJetPtMin={cfg.reco_jet_pt_min}"

    # Execute the command in the shell.
    logger.info(f"Running command: {command}")
    exit_code = os.system(command)

    # Throw an error if the command failed.
    if exit_code != 0:
        logger.error(f"Event generation failed with exit code {exit_code}.")
        raise RuntimeError(f"Event generation failed with exit code {exit_code}.")
    
    # Return.
    return