# Clustering protein sequences

## Requirements.

To use these notebooks, you only need a text editor like VSCode and Anaconda3.

These methods were developed on ubuntu 22.04.2 LTS. Other operating systems are not guaranteed.

## Installation.

On a terminal:

```
sudo apt-get install make gcc g++
conda env create -f environment.yml
```

This will create a conda environment for python 3.7, along with all the dependencies.

After opening a notebook with VScode, select the environment as the python kernel. It's called "clustering_protein_seqs".

## About notebooks.

All methods (Sequence similarity networks and non supervised learning) follows the same workflow.

First, import a dataset (in our case, a small dataset in data/ folder).

Then, makes a numerical representation of the sequences, using either one hot encoding, physicochemical properties groups, fft, or a protein language model.

After all, applies the method, either Non supervised learning, SSN or NSSN.

