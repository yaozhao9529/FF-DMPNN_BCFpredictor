from chemprop.train import cross_validate, run_training
from chemprop.args import TrainArgs
import pandas as pd
import numpy as np
import random
import torch

# Define a function to set seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Set the random seed to ensure reproducible results
set_seed(3407)

# Load the molecular descriptors feature file
features_data = pd.read_csv('../data/BCF_with_descriptors.csv')

# Display shape and preview of the features
print(f"Feature file shape: {features_data.shape}")
print(features_data.head())

# Define training arguments for Chemprop
train_args = TrainArgs().parse_args([
    '--data_path', '../data/BCF_train_data.csv',                # Path to the dataset
    '--features_path', '../data/BCF_with_descriptors.csv',      # Path to additional features
    '--dataset_type', 'regression',                             # Task type: regression
    '--save_dir', 'chemprop_checkpoints',                       # Directory to save models
    '--target_columns', 'logBCF',                               # Name of the target column
    '--smiles_column', 'SMILES',                                # Name of the SMILES column
    '--epochs', '50',                                           # Number of training epochs
    '--batch_size', '32',                                       # Batch size
    '--hidden_size', '1007',                                    # GNN hidden size
    '--depth', '2',                                             # GNN depth
    '--ffn_hidden_size', '151',                                 # Feed-forward hidden size
    '--ffn_num_layers', '1',                                    # Number of feed-forward layers
    '--dropout', '0.30703855324064994',                         # Dropout rate
    '--split_type', 'random',                                   # Random data split
    '--split_sizes', '0.8', '0.1', '0.1',                       # Train/val/test split
    '--metric', 'rmse',                                         # Primary metric
    '--extra_metrics', 'r2',                                    # Additional metrics
    '--seed', '3407',                                           # Seed for reproducibility 3407
    '--ensemble_size', '1',                                     # Number of models in ensemble
    '--num_workers', '1',                                       # Number of data loading workers
    '--no_cache_mol',                                           # Disable molecule caching
])


# Start cross-validation training using the specified arguments
cross_validate(
    args=train_args,
    train_func=run_training
)
