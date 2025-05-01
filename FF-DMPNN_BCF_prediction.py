import pandas as pd
from chemprop.train import make_predictions
from chemprop.args import PredictArgs
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import random
import torch

# Set random seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    # Set seed
    set_seed(3407)

    # Define file paths
    input_data_path = '../data/External_dataset.csv'
    features_path = '../data/External_dataset_with_descriptors.csv'
    model_path = 'chemprop_checkpoints/fold_0/model_0/model.pt'
    predictions_save_path = '../data/predictions_with_results.csv'

    # Load input data
    predict_data = pd.read_csv(input_data_path)

    # Create prediction arguments
    predict_args = PredictArgs().parse_args([
        '--test_path', input_data_path,
        '--preds_path', predictions_save_path,
        '--features_path', features_path,
        '--checkpoint_path', model_path,
    ])

    # Run Chemprop predictions
    make_predictions(args=predict_args)

    # Load prediction output
    predictions = pd.read_csv(predictions_save_path)

    # Try to identify predicted values (assumes SMILES and prediction columns)
    prediction_column = predictions.columns[-1]
    predicted_values = pd.to_numeric(predictions[prediction_column], errors='coerce')

    # Add predictions to the original data
    predict_data['predictions'] = predicted_values

    # Convert target and predictions to numeric just in case
    predict_data['logBCF'] = pd.to_numeric(predict_data['logBCF'], errors='coerce')

    # Drop rows with missing data
    valid_data = predict_data.dropna(subset=['logBCF', 'predictions'])

    # Evaluate
    r2 = r2_score(valid_data['logBCF'], valid_data['predictions'])
    rmse = np.sqrt(mean_squared_error(valid_data['logBCF'], valid_data['predictions']))

    # Output evaluation metrics
    print(f"R²: {r2}")
    print(f"RMSE: {rmse}")

    # Show a sample of the predictions
    print(predict_data)
