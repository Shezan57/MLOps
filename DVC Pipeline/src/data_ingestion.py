import os
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
import logging


# Logging configuration
logger = logging.getLogger('data_ingestion')
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('../errors.log')
file_handler.setLevel("ERROR")

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(data_url: str) -> pd.DataFrame | None:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(data_url)
        return df
    except pd.errors.ParserError as e:
        print(f"Error: Failed to parse CSV file from {data_url}")
        print(e)
    except Exception as e:
        print(f"Error: An unexpected error occurred while loading data from {data_url}")
        print(e)
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data by dropping unnecessary columns and filtering sentiment."""
    try:
        df.drop(columns=['tweet_id'], inplace=True)
        final_df = df[df['sentiment'].isin(['happiness', 'sadness'])]
        final_df['sentiment'].replace({'happiness': 1, 'sadness': 0}, inplace=True)
        return final_df
    except KeyError as e:
        print(f"Error: Missing expected column in the DataFrame: {e}")
        raise
    except Exception as e:
        print(f"Error: An unexpected error occurred while preprocessing data from {e}")
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame,data_path: str) -> None:
    try:
        data_path = os.path.join(data_path,'raw')
        os.makedirs(data_path, exist_ok=True)
        train_data.to_csv(os.path.join(data_path,"train.csv"), index=False)
        test_data.to_csv(os.path.join(data_path,"test.csv"), index=False)
    except Exception as e:
        print(f"Error: Failed to save data into {data_path}")
        print(e)
        raise

def main():
    try:
        df = load_data(data_url="https://raw.githubusercontent.com/entbappy/Branching-tutorial/refs/heads/master/tweet_emotions.csv")
        final_df = preprocess_data(df)
        train_data, test_data = train_test_split(final_df, test_size=0.2, random_state=42)
        save_data(train_data, test_data, data_path='./data')
    except Exception as e:
        print(e)
        print(f"Error: Failed to complete the data ingestion process.")

if __name__ == "__main__":
    main()