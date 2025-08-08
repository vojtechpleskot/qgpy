import qgpy.generate
from qgpy.configuration import SchedulerConfig
import concurrent.futures
import pickle
from typing import Dict, Any, List
import os

def local(cfg: SchedulerConfig, jobs: List[Dict[str, Any]], function: str):
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
        futures = [executor.submit(run_job, job) for job in jobs]
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
        job_file_name = f"{job['fit_out_dir']}/job.pkl"
        with open(job_file_name, "wb") as f:
            pickle.dump(job, f)

        # Prepare the job submission bash script.
        bash_script_name = f"{job['fit_out_dir']}/job.sh"
        with open(bash_script_name, "w") as f:
            f.write("#!/bin/bash\n")
            f.write(f"#SBATCH --partition={cfg.partition}\n")
            f.write(f"#SBATCH --job-name={histogram_name('fit', job['variables_order'], job['bin'])}\n")
            f.write(f"#SBATCH --output={job['fit_out_dir']}/slurm-%j.out\n")
            f.write(f"#SBATCH --error={job['fit_out_dir']}/slurm-%j.err\n")
            f.write(f"#SBATCH --ntasks={cfg.ntasks}\n")
            f.write(f"#SBATCH --cpus-per-task={cfg.cpus_per_task}\n")
            f.write(f"#SBATCH --time={cfg.time}\n")
            f.write(f"cd {os.getcwd()}\n")
            f.write("source venv/bin/activate\n")
            f.write(f"python {job['fit_out_dir']}/job.py\n")

        # Prepare the job submission python script.
        # It retrieves the job configuration from the pickled file and runs the run_job function.
        # This script is executed by the bash script.
        python_script_name = f"{job['fit_out_dir']}/job.py"
        with open(python_script_name, "w") as f:
            f.write("import pickle\n")
            f.write("from ufftools.submit import run_job\n")
            f.write(f"with open('{job_file_name}', 'rb') as f:\n")
            f.write("    job = pickle.load(f)\n")
            f.write("run_job(job)\n")

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
        job_file_name = f"{job['fit_out_dir']}/job.pkl"
        with open(job_file_name, "wb") as f:
            pickle.dump(job, f)

        # Prepare the job submission bash script.
        bash_script_name = f"{job['fit_out_dir']}/job.sh"
        with open(bash_script_name, "w") as f:
            f.write("#!/bin/bash\n")
            f.write(f"cd {os.getcwd()}\n")
            f.write("source venv/bin/activate\n")
            f.write(f"python {job['fit_out_dir']}/job.py\n")

        # Prepare the job submission python script.
        # It retrieves the job configuration from the pickled file and runs the run_job function.
        # This script is executed by the bash script.
        python_script_name = f"{job['fit_out_dir']}/job.py"
        with open(python_script_name, "w") as f:
            f.write("import pickle\n")
            f.write("from ufftools.submit import run_job\n")
            f.write(f"with open('{job_file_name}', 'rb') as f:\n")
            f.write("    job = pickle.load(f)\n")
            f.write("run_job(job)\n")

        # Create a submission file "job.sub"
        sub_file_name = f"{job['fit_out_dir']}/job.sub"
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

def run_job(job: Dict[str, Any]):
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

    
    # Save the fit input to a pickle file.
    with open(f"{job['run_dir']}/fit_input.pkl", "wb") as f:
        pickle.dump(fit_input, f)

    # Perform the fit.
    if job['backend'] == "zfit":
        fit_result, post_fit_hists = fit_with_zfit(hists = fit_input, templates = job['templates'], sr = job['sr'])
    elif job['backend'] == "cabinetry":
        fit_result, post_fit_hists = fit_with_cabinetry(
            hists           = fit_input,
            templates       = job['templates'],
            nf_bounds       = job['nf_bounds'],
            sr              = job['sr'],
            disc_var_label  = job['disc_var_label'],
            outdir          = job['fit_out_dir'],
        )

    # Draw the post-fit histograms.
    draw_fit(
        hists           = post_fit_hists,
        templates       = job['templates'],
        sr              = job['sr'],
        bin             = job['bin'],
        variables_order = job['variables_order'],
        variables_info  = job['variables_info'],
        templates_scale = 1,
        opts            = job['plot_opts'],
        subtext         = job['plot_opts'].subtext_postfit,
        plot_name       = f"{job['fit_out_dir']}/fit_output",
    )
    
    # Save the fit output to pickle files.
    with open(f"{job['fit_out_dir']}/fit_result.pkl", "wb") as f:
        pickle.dump(fit_result, f)
    with open(f"{job['fit_out_dir']}/post_fit_hists.pkl", "wb") as f:
        pickle.dump(post_fit_hists, f)

    # Return.
    return