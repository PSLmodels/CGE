| | |
| --- | --- |
| Org | [![PSL incubating](https://img.shields.io/badge/PSL-cataloged-a0a0a0.svg)](https://www.PSLmodels.org) [![OS License: CCO-1.0](https://img.shields.io/badge/OS%20License-CCO%201.0-yellow)](https://github.com/PSLmodels/OG-Core/blob/master/LICENSE) [![Jupyter Book Badge](https://jupyterbook.org/badge.svg)](https://pslmodels.github.io/OG-Core/) |
| Package |  [![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3108/) [![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3118/)[![Python 3.11](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3128/) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) |
| Testing | ![example event parameter](https://github.com/PSLmodels/CGE/actions/workflows/build_and_test.yml/badge.svg?branch=master) ![example event parameter](https://github.com/PSLmodels/CGE/actions/workflows/deploy_docs.yml/badge.svg?branch=master) ![example event parameter](https://github.com/PSLmodels/CGE/actions/workflows/check_black.yml/badge.svg?branch=master) [![Codecov](https://codecov.io/gh/PSLmodels/CGE/branch/master/graph/badge.svg)](https://codecov.io/gh/PSLmodels/CGE) |

# CGE

This repository contains a computational general equilibrium (CGE) model for policy analysis. The model based off of the simplest CGE model presented [Hosoe, Gawana, and Hashimoto (2010)](https://www.amazon.com/Textbook-Computable-General-Equilibrium-Modeling/dp/0230248144) and the source code is written in Python.

This work is very preliminary.  More details will be posted as development continues.

## Disclaimer

The model is currently under development. Please do not cite.

## Installing and Running CGE from this GitHub repository

* Install the [Anaconda distribution](https://www.anaconda.com/distribution/) of Python
* Clone this repository to a directory on your computer
* From the terminal (or Conda command prompt), navigate to the directory to which you cloned this repository and run `conda env create -f environment.yml`
* Then, `conda activate cge_env`
* Then install by `pip install -e .`
* Navigate to `./open_cge`
* Run the model with an example calibration by typing `python execute.py`
