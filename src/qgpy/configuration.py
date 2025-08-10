"""
configuration.py
================

This module defines dataclasses for the `hydra` package based structured
configuration management.
Type safety and schema validation is thus enforced.

Classes
-------
QGConfig
    A dataclass representing the expected configuration schema.
"""

from dataclasses import dataclass, field
from typing import Optional, Any, Dict, List
from unicodedata import name

@dataclass
class GeneralConfig:
    """
    General configuration for the qgpy package.

    Attributes
    ----------
    name : str
        The name of the experiment.
    run_dir : str
        The directory where the jobs will create their output directories and run.
        If not specified, the hydra output directory will be used.
        If specified, each job directory will be copied to the hydra output directory at the end of the run.
    generator : str
        The name of the event generator configuration to be used.
        It must match a name of one QGConfig attribute.
    """

    name            : str            = field(default = "qgpy")
    run_dir         : str            = field(default = "")
    generator       : str            = field(default = "pythia")

@dataclass
class GeneratorConfig:
    """
    Base class for the event generation configuration.

    Attributes
    ----------
    log_level : str
        The logging level for the event generation.
    """

    log_level       : str            = field(default = "INFO")
    nevents_per_job : int            = field(default = 5000)
    reco_jet_pt_min : float          = field(default = 10.0)


@dataclass
class PythiaConfig(GeneratorConfig):
    """
    Configuration of the Pythia event generator.

    Attributes
    ----------
    executable : str
        The path to the generation code executable.
        The path is relative to the main qgpy package directory.
    function : str
        The name of the function to be called for the event generation.
        There must be a function with this name in the qgpy.generate module.
    seed : int
        The random seed to be used for the Pythia 8 event generator.
    """

    executable : str            = field(default = "cpp/generate")
    function   : str            = field(default = "generate_pythia")
    seed       : int            = field(default = 0)

@dataclass
class SlicingConfig:
    """
    Configuration for the MC generation slicing.

    Attributes
    ----------
    slices_min : List[float]
        List of the minima of a slicing variable.
        E.g. the pT hat variable is used for the Pythia 8 event generator.
        Each value pairs with the corresponding slices_max value, forming a range for the slicing variable.
    slices_max : List[float]
        List of the maxima of a slicing variable.
    njobs : List[int]
        The number of jobs to be submitted for the given slicing variable range.
        Each value pairs with the corresponding slices_min and slices_max values.
        Thus, njobs[i] * nevents_per_job will be the total number of events generated
        for the i-th slicing variable range.
    """

    slices_min   : List[float]    = field(default_factory = lambda: [1000.0, 1500.0])
    slices_max   : List[float]    = field(default_factory = lambda: [1500.0, 2000.0])
    njobs        : List[int]      = field(default_factory = lambda: [1, 1])

@dataclass
class SchedulerConfig:
    """
    Base class for the job scheduler configuration.
    This class is not meant to be instantiated directly.
    It serves as a base class for the specific scheduler configurations.
    """
    pass

@dataclass
class LocalSchedulerConfig(SchedulerConfig):
    """
    Configuration of the job submission for the local machine.

    Attributes
    ----------
    max_jobs : int
        The maximum number of jobs to be submitted in parallel.
        E.g. it can be set to the number of CPU cores available on the machine.
        If set to 1, then the jobs are submitted sequentially.
        If set to -1, then all jobs are submitted at once.
    """

    max_jobs   : int            = field(default = 1)

@dataclass
class SlurmSchedulerConfig(SchedulerConfig):
    """
    Configuration of the job submission for the Slurm cluster.

    Attributes
    ----------
    command : str
        The command to be used for submitting the jobs on a cluster.
        If you are missing some options, you can add them to the command.
        E.g. you can set `command` to `sbatch --nice=10` to run the jobs with lower priority.
    partition : str
        The partition to be used for the jobs on a cluster.
    ntasks : int
        The number of tasks to be used for the jobs on a cluster.
    cpus_per_task : int
        The number of CPUs to be used for each task.
    mem : str
        The memory to be used for each task.
    time : str
        The time limit for each task.
    """

    command       : str        = field(default = "sbatch")
    partition     : str        = field(default = "ucjf")
    ntasks        : int        = field(default = 1)
    cpus_per_task : int        = field(default = 1)
    mem           : str        = field(default = "2G")
    time          : str        = field(default = "01:00:00") # 1 hour


@dataclass
class HTCondorSchedulerConfig(SchedulerConfig):
    """
    Configuration of the job scheduler.

    Attributes
    ----------
    universe: str
        The HTCondor universe type to be used for the jobs.
    should_transfer_files: str
        Whether to transfer files to the worker node.
    when_to_transfer_output: str
        When to transfer the output files.
    request_cpus: int
        The number of CPUs requested for the job.
    request_memory: str
        The memory requested for the job, e.g. "2G", "4G", etc.
    job_flavour: str
        The job flavour to be used for the HTCondor jobs.
        It is used to specify the time limit and resources for the jobs.
        The default is "espresso", which is 20 minutes.
        Other options are "microcentury" (1 hour), "longlunch" (2 hours), etc.
    """

    universe                : str            = field(default = "vanilla")
    should_transfer_files   : str            = field(default = "YES")
    when_to_transfer_output : str            = field(default = "ON_EXIT")
    request_cpus            : int            = field(default = 1)
    request_memory          : str            = field(default = "2G")
    job_flavour             : str            = field(default = "espresso")

@dataclass
class SubmitConfig:
    """
    Configuration of the job submission.

    Attributes
    ----------
    scheduler : str
        The scheduler to be used for submitting the jobs.
        Currently, only "local_parallel" is supported, which runs the jobs in parallel on the local machine.
        Another option for local execution is "local_serial", which runs the jobs sequentially.
        Supported cluster scheduler is just "slurm" at the moment.
    local : LocalSchedulerConfig
        Configuration of the job submission for the local machine.
    slurm : SlurmSchedulerConfig
        Configuration of the job submission for the Slurm cluster.
    htcondor : HTCondorSchedulerConfig
        Configuration of the job submission for the HTCondor cluster.
    """

    scheduler  : str                     = field(default = "local")
    local      : LocalSchedulerConfig    = field(default_factory = LocalSchedulerConfig)
    slurm      : SlurmSchedulerConfig    = field(default_factory = SlurmSchedulerConfig)
    htcondor   : HTCondorSchedulerConfig = field(default_factory = HTCondorSchedulerConfig)


@dataclass
class LogConfig:
    """
    Configuration of the logging.

    Attributes
    ----------
    format : str
        The logging format to be used.
    qg : str
        The logging level for the "qg" logger.
    """

    qg : str = field(default = "INFO")

@dataclass
class QGConfig:
    """
    A dataclass representing the expected configuration schema.
    """

    general                : GeneralConfig                = field(default_factory = GeneralConfig)
    pythia                 : PythiaConfig                 = field(default_factory = PythiaConfig)
    slicing                : SlicingConfig                = field(default_factory = SlicingConfig)
    submit                 : SubmitConfig                 = field(default_factory = SubmitConfig)
    log                    : LogConfig                    = field(default_factory = LogConfig)
