# [WIP] ECG Arrhythmia Classification

**Disclaimer**: This repository is work-in-progress. The code is not fully functional and documented yet.

The code was tested on Fedora 32 with Python 3.8.3. A Linux system is recommended for running this package, as it was not tested on Windows.
For training neural network models, a GPU with CUDA support is recommended.

### Setup
0. Install [Anaconda](https://www.anaconda.com/) if not yet installed.
1. Reproduce the Conda environment and activate it.
    - `conda env create -f environment_cuda.yml` for PyTorch CUDA support.
    - `conda env create -f environment_cpu.yml` for no CUDA support.
    - `conda activate ecg`
2. Run the detection.
    - `python -m detection`

### Alternative Setup
0. Install [Anaconda](https://www.anaconda.com/) if not yet installed.
1. Create a new Conda virtual environment and activate it.
    - `conda create -n ecg -c pytorch python=3.8 pytorch torchvision cudatoolkit=10.2` for PyTorch CUDA support.
    - `conda create -n ecg -c pytorch python=3.8 pytorch torchvision cpuonly` for no CUDA support.
    - `conda activate ecg`
2. Install all required Python packages.
    - `pip install -r requirements.txt`
3. Run the detection.
    - `python -m detection`
    
For all the supported parameters, you can print the help:

`python -m detection -h`


This module downloads all needed datasets, makes the train-val-test split, extracts features, trains models and evaluates them automatically. The runtime depends on the number of steps already done before.

These are approximate runtimes on Intel i7-6800K, NVIDIA GeForce RTX 2080 Ti:
- downloading data - 70 minutes
- train-val-test split - 1 minute
- feature extraction - 5 minutes
- model training - ???
- model evaluation - ???
