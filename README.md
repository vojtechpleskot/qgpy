# qgpy
Python package to generate MC events and prepare them in a format useful for AI training.

# Installation
```bash
git clone https://github.com/yourusername/qgpy.git
```

## Compile the C++ code

Setup the environment:
```bash
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
source venv/bin/activate
```

## After editing the cpp/generate.cc code
Make sure to recompile the C++ code:
```bash
cd cpp
g++ generate.cc -o generate -L/cvmfs/sft.cern.ch/lcg/views/LCG_108/x86_64-el9-gcc15-opt/lib -lpythia8 -lfastjet -L/cvmfs/sft.cern.ch/lcg/views/LCG_108/x86_64-el9-gcc15-opt/lib64/ -lHepMC3 -L../IFNPlugin/ -lIFNPlugin -I../cxxopts/include/
cd ..
```
