# Croatian Text Classification Experiments

This repository contains an end-to-end experimental pipeline for comparing classical machine-learning classifiers and a transformer-based classifier on Croatian text data.

## What the pipeline does

The script evaluates:

- TF-IDF-based classical models:
  - Random Forest
  - SVM with RBF kernel
  - Linear SVM
  - Multi-layer perceptron classifier
- A transformer-based sequence classifier using Hugging Face models
- Two text representations:
  - original text
  - lemmatized text via Stanza

The workflow produces:

- trained model artifacts
- confusion matrices
- ROC curves
- aggregated metrics in CSV and JSON format
- a summary comparison plot across models and text variants

## Main features

- Supports both original and lemmatized text preprocessing
- Includes repeated cross-validation with configurable seeds and folds
- Saves evaluation metrics in a structured format for downstream analysis
- Produces figures suitable for scientific reporting
- Exposes command-line arguments for dataset path, output directory, model name, and plotting behavior

## Requirements

Install the dependencies listed in [requirements.txt](requirements.txt).

```bash
pip install -r requirements.txt
```

## Usage

Run the script with a dataset path and an output directory:

```bash
python c2_komentari.py \
  --data-path /path/to/dataset.arrow \
  --output-dir ./outputs
```

Useful options include:

```bash
python c2_komentari.py --help
```

## Output structure

The output directory contains:

- trained model files in Joblib and Hugging Face format
- metrics files:
  - all_metrics_with_crobert.csv
  - all_metrics_with_crobert.json
- figures in the plots subfolder
- runtime summary information

## Notes for scientific review

For article review or reproducibility purposes:

- document the exact dataset version and preprocessing steps
- keep the model name and random seeds explicit
- record training hyperparameters in the repository or in a separate configuration file
- consider pinning dependency versions for full reproducibility
