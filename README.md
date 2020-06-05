# SoQal

SoQal is a method that allows a learner, in an active learning setting, to dynamically alter its dependence on an oracle for labels. 

This method is described in "SoQal: Selective Oracle Questioning in Active Learning"

# Requirements

The SoQal code requires

* Python 3.6 or higher
* PyTorch 1.0 or higher

# Datasets

## Download

The datasets can be downloaded from the following links:

1) PhysioNet 2015: https://www.physionet.org/content/challenge-2015/1.0.0/
2) PhysioNet 2017: https://physionet.org/content/challenge-2017/1.0.0/
3) Cardiology: https://irhythm.github.io/cardiol_test_set/
4) PTB: https://physionet.org/content/ptbdb/1.0.0/

## Pre-processing

In order to pre-process the datasets appropriately for SoQal, please refer to the following repository: https://anonymous.4open.science/r/9ecc66f3-e173-4771-90ce-ff35ee29a1c0/

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



