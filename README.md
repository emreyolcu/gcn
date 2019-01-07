# Graph Convolutional Network

This is a PyTorch implementation of the Graph Convolutional Network ([Kipf and
Welling, 2016](http://arxiv.org/abs/1609.02907)).

## Requirements

In order to install the required libraries it is recommended that you create a
conda environment by running

    conda env create -f environment.yaml

and before running any experiments activate the environment:

    source activate gcn

## Running

The file [`gcn.py`](gcn.py) accepts several arguments. For instance, to train a
model with the default configuration on GPU 0 you may run the command:

    python gcn.py --config_path default.yaml --gpu 0

When the `--gpu` argument is not used training is performed on the CPU.
