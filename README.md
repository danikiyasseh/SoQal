# Active Learning of Networks with SoQal

SoQal is framework that allows a network to dynamically decide, upon acquiring an unlabelled data point, whether to request a label for that data point from an oracle or to pseudo-label it instead. It can reduce a network's dependence on an oracle (e.g., physician) while maintaining its strong predictive performance. 

This repository contains a PyTorch implementation of SoQal. For details, see **SoQal: Selective Oracle Questioning for Consistency Based Active Learning of Cardiac Signals**.
[[paper](https://arxiv.org/abs/2004.09557)]

# Requirements

The SoQal code requires

* Python 3.6 or higher
* PyTorch 1.0 or higher

# Datasets

## Download

The datasets can be downloaded from the following links:

1) [PhysioNet 2015](https://www.physionet.org/content/challenge-2015/1.0.0/)
2) [PhysioNet 2017](https://physionet.org/content/challenge-2017/1.0.0/)
3) [Cardiology](https://irhythm.github.io/cardiol_test_set/)

## Pre-processing

In order to pre-process the datasets appropriately for SoQal, please refer to the following [repository](https://github.com/danikiyasseh/loading-physiological-data)

# Training

To train the model(s) in the paper, run this command:

```
python run_experiments.py
```

# Evaluation

To evaluate the model(s) in the paper, run this command:

```
python run_experiments.py
```



