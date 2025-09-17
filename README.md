# Lightweight Neural S-Box Evaluation

**This repository contains the code and resources for the work "Lightweight Neural SBox Evaluation".**

**Authors:** Binyamin Alony & Tommy Levi  
**Institution:** Bar-Ilan University

This project provides tools for generating datasets and training/testing neural networks for S-box evaluation using PyTorch.

## Getting Started

### Requirements
- Python 3.8+
- PyTorch
- numpy, tqdm, matplotlib

Install dependencies with:
```bash
pip install torch numpy tqdm matplotlib
```

## Creating a Dataset

To generate a dataset for neural network training/testing, use the `get_NN_dataset.py` script:

```bash
python get_NN_dataset.py
```

You can set parameters such as the number of samples and number of bits at the top of the script. The generated dataset will be saved as a `.pt` file in the current directory.

## Running Tests / Training the Model

To train and evaluate a neural network on your dataset, use the `testtttttttt.py` script:

```bash
python testtttttttt.py
```

This script will:
- Load the dataset
- Split it into training and test sets
- Train the model
- Print accuracy and error statistics
- Plot loss and accuracy curves

You can adjust model parameters and training settings at the top of the script.

## Notes
- Make sure the dataset file path in `testtttttttt.py` matches the file you generated.
- For more details on dataset structure and model architecture, see the comments in each script.

## File Overview
- `get_NN_dataset.py`: Script to generate and save datasets.
- `testtttttttt.py`: Script to train/test neural networks and analyze results.
- `create_NN_dataset.py`: (Optional) Additional dataset creation utilities.

---
For questions or issues, please enquire at:
- binyamin.alony@biu.ac.il
- tommy.levi@biu.ac.il