# Model Training Scripts

This directory contains scripts and utilities for training hybrid ML models for cryptocurrency price prediction.

## Structure

- `train_hybrid_models.py`: Main training script implementing walk-forward analysis
- `logs/`: Directory for training logs
- `models/`: Directory for saved model files
- `data/`: Directory for training data files

## Usage

From the repository root directory:

```bash
python -m train_models.train_hybrid_models --pair BTCEUR --window 30 --horizon 1
```

See the script documentation for detailed parameter options.