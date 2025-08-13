import shutil
import qgpy
import qgpy.utils, qgpy.generate, qgpy.convert
from qgpy.configuration import SchedulerConfig, GeneratorConfig
import concurrent.futures
import pickle
from typing import Dict, Any, List
import os

def local(cfg: SchedulerConfig, jobs: List[Dict[str, Any]]):
    """
    Submit the fit jobs in parallel on the local machine.

    Parameters
    ----------
    cfg : SchedulerConfig
        The configuration object containing the submission settings.
    jobs : List[Dict[str, Any]]
        The list of jobs to be submitted, where each job is a dictionary containing all necessary settings.

    Returns
    -------
    ret : None
        All output of each job is saved to the output directory defined in the configuration.
    """

    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers = cfg.max_jobs) as executor:
        futures = [executor.submit(run_job, **job) for job in jobs]
        results = []
        for future in futures:
            try:
                results.append(future.result())
            except Exception as exc:
                results.append(None)
                print(f"Job generated an exception: {exc}")

    # Return.
    return

def slurm(cfg: SchedulerConfig, jobs: List[Dict[str, Any]]):
    """
    Submit the fit jobs to a SLURM scheduler.

    Parameters
    ----------
    cfg : SchedulerConfig
        The configuration object containing the submission settings.
    jobs : List[Dict[str, Any]]
        The list of jobs to be submitted, where each job is a dictionary containing all necessary settings.

    Returns
    -------
    ret : None
        All output of each job is saved to the output directory defined in the configuration.
    """

    for job in jobs:

        # Pickle the job configuration.
        job_file_name = f"{job['job_dir']}/job.pkl"
        with open(job_file_name, "wb") as f:
            pickle.dump(job, f)

        # Prepare the job submission bash script.
        bash_script_name = f"{job['job_dir']}/job.sh"
        with open(bash_script_name, "w") as f:
            f.write("#!/bin/bash\n")
            f.write(f"#SBATCH --partition={cfg.partition}\n")
            f.write(f"#SBATCH --job-name={job['job_name']}\n")
            f.write(f"#SBATCH --output={job['job_dir']}/slurm-%j.out\n")
            f.write(f"#SBATCH --error={job['job_dir']}/slurm-%j.err\n")
            f.write(f"#SBATCH --ntasks={cfg.ntasks}\n")
            f.write(f"#SBATCH --cpus-per-task={cfg.cpus_per_task}\n")
            f.write(f"#SBATCH --time={cfg.time}\n")
            f.write(f"cd {os.getcwd()}\n")
            f.write("source setup.sh\n")
            f.write("unset PYTHONPATH\n")
            f.write("source venv/bin/activate\n")
            f.write(f"cd {job['job_dir']}\n")
            f.write(f"python {job['job_dir']}/job.py\n")

        # Prepare the job submission python script.
        # It retrieves the job configuration from the pickled file and runs the run_job function.
        # This script is executed by the bash script.
        python_script_name = f"{job['job_dir']}/job.py"
        with open(python_script_name, "w") as f:
            f.write("import pickle\n")
            f.write("from qgpy.submit import run_job\n")
            f.write(f"with open('{job_file_name}', 'rb') as f:\n")
            f.write("    job = pickle.load(f)\n")
            f.write("run_job(**job)\n")

        # Run the slurm job submission command.
        os.system(f"{cfg.command} {bash_script_name}")

    # Exit the program execution and tell the user to rerun after the jobs are finished.
    raise SystemExit(
        "All jobs have been submitted to the SLURM scheduler. "
        "Please wait for the jobs to finish and then rerun the script to collect the results."
        "Rerun with the same configuration and add the option hydra.run.dir=outputs/YYYY-MM-DD/HH-MM-SS/"
    )

    # Return.
    return

def htcondor(cfg: SchedulerConfig, jobs: List[Dict[str, Any]]):
    """
    Submit the fit jobs to a HTCondor scheduler.

    Parameters
    ----------
    cfg : SchedulerConfig
        The configuration object containing the submission settings.
    jobs : List[Dict[str, Any]]
        The list of jobs to be submitted, where each job is a dictionary containing all necessary settings.

    Returns
    -------
    ret : None
        All output of each job is saved to the output directory defined in the configuration.
    """


    for job in jobs:

        # Pickle the job configuration.
        job_file_name = f"{job['job_dir']}/job.pkl"
        with open(job_file_name, "wb") as f:
            pickle.dump(job, f)

        # Prepare the job submission bash script.
        bash_script_name = f"{job['job_dir']}/job.sh"
        with open(bash_script_name, "w") as f:
            f.write("#!/bin/bash\n")
            f.write(f"cd {os.getcwd()}\n")
            f.write("source venv/bin/activate\n")
            f.write(f"python {job['job_dir']}/job.py\n")

        # Prepare the job submission python script.
        # It retrieves the job configuration from the pickled file and runs the run_job function.
        # This script is executed by the bash script.
        python_script_name = f"{job['job_dir']}/job.py"
        with open(python_script_name, "w") as f:
            f.write("import pickle\n")
            f.write("from qgpy.submit import run_job\n")
            f.write(f"with open('{job_file_name}', 'rb') as f:\n")
            f.write("    job = pickle.load(f)\n")
            f.write("run_job(**job)\n")

        # Create a submission file "job.sub"
        sub_file_name = f"{job['job_dir']}/job.sub"
        rel_dir_name = os.path.relpath(bash_script_name, start=os.getcwd())
        with open(sub_file_name, "w") as f:
            f.write(f"executable = {rel_dir_name}\n")
            f.write(f"universe = {cfg.universe}\n")
            f.write(f"should_transfer_files = {cfg.should_transfer_files}\n")
            f.write(f"when_to_transfer_output = {cfg.when_to_transfer_output}\n")
            f.write(f"error = {rel_dir_name}/job.err\n")
            f.write(f"output = {rel_dir_name}/job.out\n")
            f.write(f"log = {rel_dir_name}/job.log\n")
            f.write(f"request_cpus = {cfg.request_cpus}\n")
            f.write(f"request_memory = {cfg.request_memory}\n")
            f.write(f"queue\n")
            f.write(f"+JobFlavour = \"{cfg.job_flavour}\"\n")




        # Run the HTCondor job submission command.
        os.system(f"condor_submit {sub_file_name}")

    # Exit the program execution and tell the user to rerun after the jobs are finished.
    raise SystemExit(
        "All jobs have been submitted to the HTCondor scheduler. "
        "Please wait for the jobs to finish and then rerun the script to collect the results."
        "Rerun with the same configuration and add the option hydra.run.dir=outputs/YYYY-MM-DD/HH-MM-SS/"
    )


    # Return.
    return

def run_job(
        job_name: str,
        job_dir: str,
        target_dir: str,
        cfg: GeneratorConfig,
        delphes_card: str,
        slice_min: float = -1,
        slice_max: float = -1,
        copy_output_on_exit: bool = False,
        ) -> None:
    """
    Run the job with the given configuration.

    Parameters
    ----------
    job : Dict[str, Any]
        The job configuration dictionary containing all necessary settings for the job.

    Returns
    -------
    ret : None
        The output of the job is saved to the output directory defined in the job configuration.
    """

    # Uncomment the following lines if you want to use matplotlib for plotting in this function!
    # # Avoid X server issues in non-GUI environments.
    # matplotlib.use('Agg')

    # Create the job logger.
    logger = qgpy.utils.create_logger(job_name, outdir = job_dir)
    logger.info("Beginning of the run_job function...")


    # Get the generate function to call.
    logger.info(f"Getting the {cfg.function} function from the qgpy.generate module...")
    gen_func = getattr(qgpy.generate, cfg.function)

    # Call the generate function.
    logger.info(f"Calling the {cfg.function} function...")
    gen_func(
        outdir=job_dir,
        cfg=cfg,
        slice_min=slice_min,
        slice_max=slice_max
    )


    # Run Delphes on the hepmc3 file, unless delphes.root exists.
    delphes_file = f"{job_dir}/delphes.root"
    if os.path.exists(delphes_file):
        logger.info(f"Delphes file {delphes_file} already exists. Skipping Delphes run.")
    else:
        package_dir = os.path.dirname(os.path.abspath(qgpy.__file__))
        logger.info(f"Running Delphes on the generated HepMC3 file using the {package_dir}/../../{delphes_card} card.")
        logger.info(f"Delphes file will be saved to: {delphes_file}")
        os.system(f'DelphesHepMC3 {package_dir}/../../{delphes_card} {job_dir}/delphes.root {job_dir}/generate.hepmc3')

    # # Convert the Delphes root format to the JIDENN accepted root format.
    # qgpy.convert.delphes_to_jidenn_root(
    #     delphes_file=f"{job_dir}/delphes.root",
    #     jidenn_file=f"{job_dir}/jidenn_input.root"
    # )


    # Convert the Delphes root format to the TensorFlow dataset format.
    dataset_dir = f"{job_dir}/tf_dataset"
    if not os.path.exists(dataset_dir):
        qgpy.convert.delphes_to_tf_dataset(
            job_dir=job_dir,
            delphes_file=f"{job_dir}/delphes.root",
            dataset_dir=dataset_dir,
        )









    # At the end of the job, copy the directory job_dir to the target_dir,
    # unless job_dir is subdirectory of target_dir.
    # Note: copy the whole directory, not just the contents.
    if copy_output_on_exit:
        if not os.path.commonpath([os.path.abspath(job_dir), os.path.abspath(target_dir)]) == os.path.abspath(target_dir):
            dest = os.path.join(target_dir, os.path.basename(job_dir))
            shutil.copytree(job_dir, dest, dirs_exist_ok=True)

    # Return.
    return