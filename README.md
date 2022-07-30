# COVID-19 SEIRA Model Parameter Estimation with Neural Network and Numerical ODE Solver


## Introduction
Extracting parameters of SEIRA model for COVID-19 by using odeint-augmented machine learning process.

input data is daily new positive/death cases for 30 days and output be estimations for 9 states of SEIRA model for 60 days.


## Project Configuration
* data: directory for processed COVID-19 data for training (*.npz file)
* out: directory for saving training results (*.pt and figures)
* src
  - dev: prototypes
  - util/covid_data_preprocessor.py: produce *.npz file for training
  - model1
    * Dataset.py: dataset class for training/validation
    * Network.py: ParameterNet, SEIRA IVP module
    * train.py: run machine learning process
    * util.py: dataloader, helper functions are implemented



## Installation
### Create virtualenv
```shell
$ virtualenv cmp -p python3.8
$ source cmp/bin/activate
```

Deactivate virtualenv with
```shell
$ deactivate
```

### install requirements
```shell
$ pip install -r requirements.txt
```

## training
```shell
$ python src/model1/train.py
```

