# qgpy
Python package to generate MC events and prepare them in a format useful for AI training.

ALL INSTRUCTION ARE GIVEN FOR RUNNING JUST ON THE CHIMERA CLUSTER, FOR THE MOMENT.

# Installation

## Download the repository
```bash
git clone https://github.com/vojtechpleskot/qgpy.git
cd qgpy/
git submodule update --init --recursive
```
Even better, you can fork the repository into your GitHub account and clone your forked repository.

## Compile the C++ code

Setup the environment:
```bash
cd <path_to_the_qgpy_directory>
source setup.sh
```

Compile the IFNPlugin:
```bash
cd IFNPlugin
make -j
cd ..
```

Compile the event generation code:
```bash
cd cpp
g++ generate.cc -o generate -L/cvmfs/sft.cern.ch/lcg/views/LCG_108/x86_64-el9-gcc15-opt/lib -lpythia8 -lfastjet -L/cvmfs/sft.cern.ch/lcg/views/LCG_108/x86_64-el9-gcc15-opt/lib64/ -lHepMC3 -L../IFNPlugin/ -lIFNPlugin -I../cxxopts/include/
cd ..
```

## Install the Python package
Make sure to unset the `PYTHONPATH` to avoid conflicts with python packages installed in the LCG environment:
```bash
unset PYTHONPATH
python -m venv venv
source venv/bin/activate
pip install -e .
```

## On the next startup
Make sure to activate the LCG and the virtual environments again:
```bash
cd <path_to_the_qgpy_directory>
source setup.sh
unset PYTHONPATH
source venv/bin/activate
```

## After editing the cpp/generate.cc code
Make sure to recompile the C++ code:
```bash
cd cpp
g++ generate.cc -o generate -L/cvmfs/sft.cern.ch/lcg/views/LCG_108/x86_64-el9-gcc15-opt/lib -lpythia8 -lfastjet -L/cvmfs/sft.cern.ch/lcg/views/LCG_108/x86_64-el9-gcc15-opt/lib64/ -lHepMC3 -L../IFNPlugin/ -lIFNPlugin -I../cxxopts/include/
cd ..
```

# Configuration
The configuration is done with the `qgpy/yaml_config/config.yaml` file.
The file is processed with the [Hydra](https://hydra.cc/docs/intro/) framework, which allows for easy configuration management and overrides from the command line.
Just very briefly:
- Check which parameter you want to change. All parameters are in the dataclasses defined in `qgpy/configuration.py`.
- To change e.g. 

# JIDENN
This package is a framework for training neural networks on jet data, particularly focusing on the use of quark-gluon jet tagging.
It provides a variety of neural network architectures and utilities for data preprocessing, training, and evaluation.
The package is [here](https://github.com/vojtechpleskot/JIDENN.git)

## Installation
To install the package, clone the repository, run the tensorflow container, create the python virtual environment, and install the requirements:
```bash
git clone https://github.com/vojtechpleskot/JIDENN.git
cd JIDENN
#apptainer run --bind=/home --bind=/work --bind=/scratch --bind /singularity/ucjf:/singularity_ucjf --nv /home/jankovys/tensorflow_latest-gpu.sif
python -m venv venv
source venv/bin/activate
pip install tensorflow[and-cuda]==2.18.0
pip install -r requirements_no_version.txt
```

## On the next startup
Make sure to activate the environment again:
```bash
cd <path_to_the_JIDENN_directory>
#apptainer run --bind=/home --bind=/work --bind=/scratch --bind /singularity/ucjf:/singularity_ucjf --nv /home/jankovys/tensorflow_latest-gpu.sif
source venv/bin/activate
```

## How to use JIDENN using qgpy-generated data

1. Generate the data using the `qgpy` package as described in the qgpy documentation above.

2. Open new interactive job.

3. Setup the environment, see [above](#on-the-next-startup-1).

4. Concatenate the tf datasets from qgpy. Run commands similar to:

```bash
d=/scratch/ucjf-atlas/plesv6am/qg/data21  # Path where the qgpy-generated data is stored
python scripts/combine_files.py --load_path $d --start_identifier slice --save_path $d/tf_dataset_combined --subdir tf_dataset
```
   - `slice` is the beginning of subdirectory names in the directory `$d`.
   - `tf_dataset` is the name of the subdirectory in each `slice*` directory.
   - NOTE: The `slice*` and `tf_dataset` names are the `qgpy` package defaults.

5. Flatten the events to jets and flatten the jet spectrum in one go:

```bash
python scripts/flatten_spectrum.py --load_path $d/tf_dataset_combined/train/ --save_path $d/tf_dataset_flatten --pt_lower_cut 200 --pt_upper_cut 300 --eta_cut 2.1 --flattening_var jets_pt jets_eta --bins 10 10
```
   - `--pt_lower_cut` and `--pt_upper_cut` define the pt range of jets to keep.
   - You can also set the `--eta_cut` parameter to limit the jet absolute eta range. Default is `2.1`.
   - `--flattening_var` specifies which variables to use for flattening the jet spectrum.
   - `--bins` specifies the number of bins for each flattening variable.
   - You might need to run the flattening script multiple times: for the `train/`, `dev/`, and `test/` subdirectories separately.

6. Create and/or edit the `jidenn/yaml_config/config_test.yaml` file. The file `/home/plesv6am/qg/JIDENN/jidenn/yaml_config/config_test.yaml` should work - you can use it as a template; copy it to `jidenn/yaml_config/`.

   - Make sure to set the `data.path`, `data.dev_path`, and `test_data.path` to the path where you saved the flattened data.

7. Start the training with the `jidenn/train.py` script.
NOTE: You need to run the training script on a node with GPUs, otherwise it will likely not work.
To log there, start from a clean hpc session, and start the following interactive job:
```bash
salloc -p gpu-ffa --cpus-per-task 12 --mem 100G --time=12:00:00 --gres=gpu
```
Then, setup the environment as above, and run:
```bash
python train.py --config-name config_test
```
