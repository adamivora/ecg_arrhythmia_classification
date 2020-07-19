# ECG Arrhythmia Classification

A comparison of various machine learning models on the task of classifying heart arrhythmia from 1-lead ECG on publicly available datasets ([CINC2017](https://physionet.org/content/challenge-2017/1.0.0/) and [CPSC2018](http://2018.icbeb.org/Challenge.html)).

Implemented for my bachelor's thesis [ECG Arrhythmia Detection and Classification](https://github.com/adamivora/ecg_arrhythmia_classification/blob/master/thesis/ECG_Arrhythmia_Detection_and_Classification.pdf) at Faculty of Informatics, Masaryk University.    

### Hardware + software requirements
The code was tested on Fedora 32 with Python 3.8.3. A Linux system is recommended for running this package, as it was not tested on Windows.
For training neural network models, a GPU with CUDA support and at least 8 GB of GPU RAM is recommended. The detection uses around 14 GB RAM, so at least 24 GB are recommended.  

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


This module downloads all needed datasets, makes the train-val-test split, extracts features, trains models and evaluates them automatically. The runtime depends on the number of steps already done.

These are approximate runtimes on Intel i7-6800K, NVIDIA GeForce RTX 2080 Ti:
- downloading data: 70 minutes
- train-val-test split: 1 minute
- feature extraction: 30 minutes
- model training: 2-3 hours
- model evaluation: 1 minute

### Used libraries
This work is based on several machine learning, physiological data processing and data science Python libraries.
Kudos to all these projects for providing the functionality I used:
- [NeuroKit2](https://github.com/neuropsychology/NeuroKit)
- [ONNXMLTools](https://github.com/onnx/onnxmltools)
- [pandas](https://github.com/pandas-dev/pandas)
- [plotly.py](https://github.com/plotly/plotly.py)
- [PyTorch](https://github.com/pytorch/pytorch)
- [scikit-learn](https://github.com/scikit-learn/scikit-learn)
- [SciPy](https://github.com/scipy/scipy)
- [tqdm](https://github.com/tqdm/tqdm)
- [tslearn](https://github.com/tslearn-team/tslearn)
- [XGBoost](https://github.com/dmlc/xgboost)
- and to all the others I forgot to mention.
