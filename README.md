# brf-neurons-reproduction
This repository consists of a partial reproduction and extension of the experiments from the paper:

> **Balanced Resonate-and-Fire Neurons**
> Saya Higuchi, Sebastian Kairat, Sander M. Bohté, Sebastian Otte
> *Proceedings of the 41st International Conference on Machine Learning (ICML), 2024*

## Overview

The Balanced Resonate-and-Fire (BRF) neuron introduces stable and sparse oscillatory dynamics into spiking neural networks (SNNs), outperforming previous spiking models such as ALIF and standard RF neurons in both performance and efficiency.

## Python Version

This project requires **Python 3.10.4**, which was used in the original experiments.

## Installation

Create a new virtual environment and install the required dependencies:

```bash
# Clone the repository
git clone https://github.com/ianjoshi/brf-neurons-reproduction.git
cd brf-neurons-reproduction

# Create virtual environment
python3.10 -m venv brf-venv
source brf-venv\Scripts\activate  # On Windows

# Install required packages
pip install -r requirements.txt
```

## Reproduction of Figure 5
This repository reproduces the experiments corresponding to Figure 5 of the original BRF paper. These experiments benchmark BRF neurons against standard RF and ALIF models across multiple sequential classification tasks using SNNs. All reproduction scripts and results are located in the `experiments/` directory. The code structure and experimental setup are aligned with the original BRF repository: https://github.com/AdaptiveAILab/brf-neurons, with minor adaptations for local reproducibility. The following datasets were used:
- SMNIST (Sequential MNIST)
- PMNIST (Permuted Sequential MNIST)
- ECG (Electrocardiogram classification)
- SHD (Spiking Heidelberg Digits)


## Citation

If you use this code, please cite the original paper:

```
@inproceedings{higuchi2024balanced,
  title={Balanced Resonate-and-Fire Neurons},
  author={Higuchi, Saya and Kairat, Sebastian and Bohte, Sander M and Otte, Sebastian},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2024}
}
```

Also consider referencing the official implementation repository:
```
https://github.com/AdaptiveAILab/brf-neurons
```