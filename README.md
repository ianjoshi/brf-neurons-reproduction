# brf-neurons-reproduction
This repository consists of a partial reproduction and extension of the experiments from the paper:

> **Balanced Resonate-and-Fire Neurons**
> Saya Higuchi, Sebastian Kairat, Sander M. Bohté, Sebastian Otte
> *Proceedings of the 41st International Conference on Machine Learning (ICML), 2024*

## Overview

The Balanced Resonate-and-Fire (BRF) neuron introduces stable and sparse oscillatory dynamics into spiking neural networks (SNNs), outperforming previous spiking models such as ALIF and standard RF neurons in both performance and efficiency.

## Python Version

This project requires **Python 3.10.4**, which was used in the original experiments.

## Setup Instructions

Create a new virtual environment and install the required dependencies:

```bash
# Clone the repository
git clone https://github.com/ianjoshi/brf-neurons-reproduction.git
cd brf-neurons-reproduction

# Create virtual environment
python3.10 -m venv brf-venv
brf-venv\Scripts\activate  # On Windows

# Install required packages
pip install -r requirements.txt
```

## Reproduction of Figure 5
This repository reproduces the experiments corresponding to Figure 5 of the original BRF paper. These experiments benchmark BRF neurons against standard RF and ALIF models across multiple sequential classification tasks using SNNs. All reproduction scripts and results are located in the `experiments/` directory. The code structure and experimental setup are aligned with the original BRF repository: https://github.com/AdaptiveAILab/brf-neurons, with minor adaptations for local reproducibility. The following datasets were used:
- SMNIST (Sequential MNIST)
- PMNIST (Permuted Sequential MNIST)
- ECG (Electrocardiogram classification)
- SHD (Spiking Heidelberg Digits)


## Experiments with Google Speech Commands
We extend the BRF neuron experiments to the Google Speech Commands (GSC) dataset to evaluate their performance on a real-world audio keyword classification. We use the Google Speech Commands v0.03 dataset, as distributed via TensorFlow Datasets: https://www.tensorflow.org/datasets/catalog/speech_commands.

All code specific to this extension is located in the `gsc/` directory. This includes:
- `gsc/data/`:
  - `SpeechCommands.py`: Custom PyTorch Dataset class that wraps the official `torchaudio.datasets.SPEECHCOMMANDS`.
  - `SpeechCommandsDataLoader.py`: Encapsulates logic to build `DataLoaders` for all three splits (train, val, test). It:
    - Initializes MFCC transforms and one-hot label encodings.
    - Instantiates `SpeechCommands` objects with caching, sequence length, and sampling percentage options.
    - Returns PyTorch `DataLoaders` for each split.
  - `OneHotTargetTransform.py`: Utility class that converts a string label into a one-hot tensor and repeats that one-hot vector over a fixed number of time steps, to match the input sequence length.
  - `Preprocessor.py`: Formats input-output pairs into the correct shape for training.
- `gsc/train/`:
  - `gsc_brf_train.py`: Trains the BRF model on GSC.
  - `gsc_rf_train.py`: Trains the RF model on GSC.
  - `gsc_alif_train.py`: Trains the ALIF model on GSC.

Initial and trained models and evaluation plots are saved in the `gsc-experiments/` directory. Note that TensorBoard logs are excluded from this due to storage constraints.

### GSC Setup Instructions
Create a new GSC-specific virtual environment and install the GSC-specific dependencies:
```bash
python3.10 -m venv gsc-venv
gsc-venv/Scripts/activate  # Windows:
pip install -r gsc-experiments/gsc-requirements.txt
```

### Running Experiments
To run an experiment:
```bash
python gsc/train/gsc_brf_train.py   # BRF model
python gsc/train/gsc_rf_train.py    # RF model
python gsc/train/gsc_alif_train.py  # ALIF model
```

## Experiments with Energy aware loss

We extend the BRF neuron experiments specifically on the SMNIST dataset by adding an energy aware loss. The code specific to this extension can be found in the `experiments` directory under:

- `experiments/`
  - `smnist/`
    - `smnist_train_spike_loss.py`
  - `SOP_inspection.ipynb`

The `smnist_train_spike_loss.py` works exactly the same as the `smnist_train.py` file, and the runs and models are stored at the same locations. In this file, the `lambda_spike` parameter can be tuned to explore different loss values. 

The file `SOP_inspection.ipynb` can be used to inspect the energy efficiency of a model. 

To run the experiment:
```bash
python experiments/smnist/smnist_train_spike_loss.py
```

## Experiments with Linear Decay

We extend the BRF neuron experiments specifically on the SHD and ECG datasets by changing the decay of neuron resonance from exponential to linear. The code specific to this extension can be found in the `snn` directory under:

- `snn/`
  - `modules_linear/`

In order to run this code, the same files that are used to run standard SHD and ecg can be used, with one caveat: in the file /models/resonaternns.py, the line
'''from .. import modules'''
must be replaced with the line 
'''from .. import modules_linear as modules'''

To run the experiment with the SHD dataset:
```bash
python experiments/SHD/shd_train.py
python experiments/SHD/shd_alif_train.py
```

To run the experiment with the ECG dataset:
```bash
python experiments/ecg/ecg_train.py
python experiments/ecg/ecg_alif_train.py
```

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
