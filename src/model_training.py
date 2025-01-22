import os
import numpy as np
import pandas as pd
import pickle
import logging
from sklearn.ensemble import RandomForestClassifier
import yaml

# Ensure log directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# logging configuration
logger = logging.getLogger('model_training')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'model_training.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter("%(asctime)s -%(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(params_path: str) -> dict:
    """
    Load the parameters from the specified YAML file.
    """
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug("Parameters loaded from %s", params_path)
        return params
    
    except FileNotFoundError:
        logger.error("Parameters file not found: %s", params_path)
        raise
    
    except yaml.YAMLError as e:
        logger.error("Failed to parse YAML file: %s", e)
        raise
    
    except Exception as e:
        logger.error("Unexpected error occured while loading parameters: %s", e)
        raise
    


# load data
def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from csv file
    :param file_path: path to csv file
    :return: pandas DataFrame
    """
    try:
        df = pd.read_csv(file_path)
        logger.debug("Data loaded from %s", file_path, df.shape)
        return df
    
    except pd.errors.ParserError as e:
        logger.error("Failed to parse the csv file: %s", e, exc_info=True)
        raise
    
    except FileNotFoundError as e:
        logger.error("Failed to find the file: %s", e)
        raise
    
    except Exception as e:
        logger.error("Unexpected error occurred while loading the data: %s", e)
        raise
    
    
def train_model(X_train: np.ndarray, y_train: np.ndarray, params: dict) -> RandomForestClassifier:
    """
    Train Random Forest model
    :param X_train: features for training
    :param y_train: target for training
    :param params: hyperparameters for Random Forest model
    :return: trained RandomForestClassifier model
    """
    try:
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("X_train and y_train should have the same number of samples")
        
        logger.debug("Initializing RandomForest Classifier with parameters: %s", params)
        
        clf = RandomForestClassifier(n_estimators=params['n_estimators'], random_state=params['random_state'])
        
        logger.debug("Model started training with %d samples", X_train.shape[0])
        
        clf.fit(X_train, y_train)
        logger.debug("Model training completed successfully")
        
        return clf
    
    except ValueError as e:
        logger.error("Invalid input: %s", e)
        raise
    
    except Exception as e:
        logger.error("Unexpected error occurred while training the model: %s", e)
        raise
    
    
def save_model(model, file_path: str) -> None:
    """ 
    Save the model to a file.
    
    :param model: trained model
    :param file_path: path to save the model
    """
    try:
        # ensure that directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
            
        logger.debug("Model saved to %s", file_path)
        
    except FileNotFoundError as e:
        logger.error("Failed to find the directory: %s", e)
        raise
    
    except Exception as e:
        logger.error("Unexpected error occurred while saving the model: %s", e)
        raise
    

def main():
    
    try:
        
        params = load_params('params.yaml')['model_training']
        # params = {'n_estimators':25, 'random_state': 2}
        
        # load data
        train_data = load_data('./data/processed/train_tfidf.csv')
        
        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values
        
        clf = train_model(X_train, y_train, params)
        
        save_model_path = 'models/model.pkl'
        save_model(clf, save_model_path)
        
    except Exception as e:
        logger.error("Failed to complete the feature engineering pipeline: %s", e)
        print(f"Error: {e}")
        

if __name__ == "__main__":
    main()