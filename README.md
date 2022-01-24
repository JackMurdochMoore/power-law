# Introduction
This repository contains the codes to generate different types of constrained (and other) power-law surrogates. It allows you to

- generate constrained surrogates based on a time series.
- reproduce the results of the manuscript *Non-parametric power-law surrogates", by Jack. Murdoch Moore, Gang Yan, and Eduardo G. Altmann*


# How-to use


A tutorial to generate surrogates based on a new or existing time series is given in the [Jupyter notebook](https://jupyter.org/) tutorial.ipynb:

'''
jupyter-notebook tutorial.ipynb
'''

In order to reproduce the results fo the manuscript, you should run the notebook 'generating-results.ipynb' (to generate the results) and the notebook 'make-figures.ipynb' (to generate the figures).


# Organization of the repository:

## Folders

- time series: contains the data used in this repository.
- figures: contains the figures shown in the manuscript
- results: stores the outputs of the calculations

## Files

- 'constrained_power_law_surrogates.py': source codes.
- 'requirements.txt': python packages required in the repository.