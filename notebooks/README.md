# Introduction
This folder contains Jupyter notebooks for associated with [this repository](https://github.com/JackMurdochMoore/power-law).

- generate constrained surrogates based on a time series
- perform simple hypothesis tests with constrained surrogates
- reproduce the results of the manuscript *Non-parametric power-law surrogates", by Jack Murdoch Moore, Gang Yan, and Eduardo G. Altmann*


# How-to use


A tutorial to generate surrogates based on a new or existing time series is given in the [Jupyter notebook](https://jupyter.org/) 'tutorial.ipynb' in the folder 'notebooks':

'''
jupyter-notebook tutorial.ipynb
'''

In order to reproduce the results of the manuscript, you should run the notebook 'generate-results.ipynb' with the parameters of the manuscript (to generate the results) and the notebook 'make-figures.ipynb' (to generate the figures). Both 'generate-results.ipynb' and 'make-figures.ipynb' are in the folder 'notebooks'.


# Organization of this folder:

## Files

- 'tutorial.ipynb': A tutorial to generate surrogates based on a new or [existing time series](https://github.com/JackMurdochMoore/power-law/tree/main/time-series).
- 'generate-results.ipynb': A notebook to reproduce the results of the manuscript based on a new or [existing time series](https://github.com/JackMurdochMoore/power-law/tree/main/time-series) and store them in the folder [results](https://github.com/JackMurdochMoore/power-law/tree/main/results).
- 'make-figures.ipynb': A notebook to produce most figures of the manuscript based on results in the folder [results](https://github.com/JackMurdochMoore/power-law/tree/main/results) and save them in the folder [figures](https://github.com/JackMurdochMoore/power-law/tree/main/figures).
