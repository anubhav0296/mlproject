import os
import sys

from components.data_transformation import DataTransformation
from components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# Any input that is required, will be given in this function
# this decorator is used, inside a class to define a variable
# we use init, but inside a class, to define a class variable

@dataclass # This helps defining directly the class variable
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "data.csv")
    train_data_path: str = os.path.join("artifacts", "train_data.csv")
    test_data_path: str = os.path.join("artifacts", "test_data.csv")

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig
    
    def initialize_data_ingestion(self):
        try:
            logging.info("Data Ingestion has been initialized")

            # First read the raw file
            df = pd.read_csv('notebook/data/stud.csv')

            # Then create the raw folder
            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_path), exist_ok=True)

            # Once the raw folder path exists, export the csv to that particular path
            df.to_csv(self.data_ingestion_config.raw_data_path, index=False, header=True)

            # Create folders for both train and test paths
            os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.data_ingestion_config.test_data_path), exist_ok=True)

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.data_ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.test_data_path, index=False, header=True)

            logging.info("All the ingestion is now completed")

            return (
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initialize_data_ingestion()

    data_transformation_obj = DataTransformation()
    train_arr, test_arr, _ = data_transformation_obj.initiate_data_transformation(train_data, test_data)

    # model_trainer_obj = ModelTrainer()
    # print(f"Best model r2 score  is - {model_trainer_obj.initiate_model_trainer(train_arr,test_arr)}")

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))





