# Introduction
This repository contains the codes to generate different types of constrained (and other) power-law surrogates. It allows you to

- generate constrained surrogates based on a time series
- perform simple hypothesis tests with constrained surrogates
- reproduce the results of the manuscript *Non-parametric power-law surrogates*, by Jack Murdoch Moore, Gang Yan, and Eduardo G. Altmann


# How-to use


A tutorial to generate surrogates based on a new or existing time series is given in the [Jupyter notebook](https://jupyter.org/) 'tutorial.ipynb' in the current folder:

'''
jupyter-notebook tutorial.ipynb
'''

In order to reproduce the results of the manuscript, you should run the notebook 'generate-results.ipynb' with the parameters of the manuscript (to generate the results) and the notebook 'make-figures.ipynb' (to generate the figures). Both 'generate-results.ipynb' and 'make-figures.ipynb' are in the folder [reproduce-paper](https://github.com/JackMurdochMoore/power-law/tree/main/reproduce-paper).


# Organization of the repository:

## Folders

- src: contains source code (i.e., the module 'constrained_power_law_surrogates.py')
- time series: contains the data used in this repository
- results: stores the outputs of the calculations
- reproduce_paper: code and figures that reproduce results of the manuscript


## Files

- 'requirements.txt': python packages required in the repository.
- 'tutorial.ipynb': A tutorial to generate surrogates based on a new or [existing time series](https://github.com/JackMurdochMoore/power-law/tree/main/time-series).