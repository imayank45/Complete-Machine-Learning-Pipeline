import os
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
import nltk
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')


# Ensure log directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# logging configuration
logger = logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'data_preprocessing.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter("%(asctime)s -%(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def transform_text(text):
    """Transform the input text by converting it to a lowercase, tokenizing,
    removing stopwords, punctuation and stemming
    """
    ps = PorterStemmer()

    # convert to lowercase
    text = text.lower()
    
    # tokenize the text
    text = nltk.word_tokenize(text)
    
    # remove non-alphanumeric tokens
    text = [word for word in text if word.isalnum()]
    
    # remove stopwords and punctuation
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    
    # stemming
    text = [ps.stem(word) for word in text]
    
    return ' '.join(text)


def preprocess_df(df, text_column = 'text', target_column = 'target'):
    """Preprocess the data frame by encoding a column, removing duplicates and 
    transforming the text column
    """
    try:
        logger.debug('Starting preprocessing for dataframe')
                     
        # encode target column
        encoder = LabelEncoder()
        df[target_column] = encoder.fit_transform(df[target_column])
        logger.debug('Target column encoded')
        
        # remove duplicates
        df = df.drop_duplicates(keep='first')
        logger.debug('Duplicates removed')
        
        # apply text transformation to the specified text column
        df.loc[:, text_column] = df[text_column].apply(transform_text)
        logger.debug('Text column transformation applied')
        return df
        
    except KeyError as e:
        logger.error("Missing column in the dataframe: %s", e)
        raise
        
    except Exception as e:
        logger.error("Unexpected error during preprocessing: %s", e)
        raise
    
    
def main(text_column='text', target_column='target'):
    try:
        """Main function to load raw data, preprocess it 
        and save the processed data
        """
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logger.debug("Data loaded properly")
        
        # transform data
        train_data_processed = preprocess_df(train_data, text_column, target_column)
        test_data_processed = preprocess_df(test_data, text_column, target_column)
        
        # store the data inside data/processed
        data_path = os.path.join("./data", "interim")
        os.makedirs(data_path, exist_ok=True)
        
        train_data_processed.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
        test_data_processed.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)
        
        logger.debug("Processed data saved to %s", data_path)
    
    except FileNotFoundError as e:
        logger.error("Failed to find the data file: %s", e)
        
    
    except pd.errors.EmptyDataError as e:
        logger.error("No data file found: %s", e)
        
    except Exception as e:
        logger.error("Failed to complete the data transformation process: %s", e)
        print(f"Error: {e}")
        
        
if __name__ == "__main__":
    main()