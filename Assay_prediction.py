import pandas as pd
from chemprop.train import make_predictions
from chemprop.args import PredictArgs

if __name__ == '__main__':
    # 1. Load the input data for prediction
    prediction_data_path = '../data/BCF_model_assay_dataset.csv'
    data = pd.read_csv(prediction_data_path)

    # 2. Define prediction arguments, pointing to the saved model directory
    # Replace 'CAR' with other target names (e.g., 'PPARd') as needed
    predict_args = PredictArgs().parse_args([
        '--test_path', prediction_data_path,               # Path to test/prediction data
        '--checkpoint_dir', 'chemprop_checkpoints/CAR/fold_0',  # Trained model directory
        '--preds_path', 'assay_predictions.csv',       # Output path for prediction results
        '--smiles_columns', 'SMILES',                  # Name of the SMILES column in CSV
    ])

    # 3. Run predictions using the trained Chemprop model
    make_predictions(args=predict_args)

    # 4. Print confirmation message
    print("预测完成，结果保存在:", predict_args.preds_path)
