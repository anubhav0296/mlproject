import os
import sys
from dataclasses import dataclass

from utils import save_object

from src.exception import CustomException
from src.logger import logging

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

import numpy as np
import pandas as pd


@dataclass
class DataTransformationConfig:
    # Once the Data Transformation is done, the output file will be stored in below path
    pre_processor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    # This function helps in assigning an object to DataTransformationConfig
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig

    ''' This function is responsible for getting preprocessor object that does all the data transformation'''
    # This function mainly gets what what tranformation needs to be done on the data
    def get_data_transformation_object(self):
        try:
            logging.info("Getting data transformation object started")

            # Segregating the columns and storing them in different variables (Numerical and Categorical)
            num_cols = ["reading_score","writing_score"]

            cat_cols = ["gender","race_ethnicity","parental_level_of_education","lunch","test_preparation_course"]

            # Passing the steps in Pipeline what what Transformation needs to be done (For Numerical Data)
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler(with_mean=False))
                ])
            
            # Categorical Column - Mode Imputation (Missing), OneHotEncoding (Cat --> Num), Standardizing
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ])

            logging.info("Numerical columns transformation is done")
            logging.info("Categorical column tarnsformation is done")

            # Here we just pass the pipeline and columns, ColumnTransformers transforms the data
            # and we store the Output in pre_processor variable
            pre_processor = ColumnTransformer(
                [
                ("num_pipeline", num_pipeline, num_cols),
                ("cat_pipeline", cat_pipeline, cat_cols)
            ])

            logging.info("Preprocessing is completed")

            return pre_processor
        
        except Exception as e:
            raise CustomException(e, sys)
    
    # This functions takes the input data and ingests to pre_processor, which holds all the 
    # steps to be followed for data transformation and then finally creates 3 outputs
    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Reads train and test data from path and stores in a df 
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # Bifurcate the input features and target variables from train and test df
            target_col = ["math_score"]

            input_feature_train_df = train_df.drop(target_col, axis=1)
            target_feature_train_df = train_df[target_col]

            input_feature_test_df = test_df.drop(target_col, axis=1)
            target_feature_test_df = test_df[target_col]

            # Get the data transformation O/P and store it in a variable to create an object
            pre_processor_obj = self.get_data_transformation_object()

            # Fit train and test data in pre_processor object
            pre_processed_train_df = pre_processor_obj.fit_transform(input_feature_train_df)
            pre_processed_test_df = pre_processor_obj.transform(input_feature_test_df)

            # Concatenates the processed train dataframe and target variable (To create an arr)
            train_arr = np.c_[pre_processed_train_df, np.array(target_feature_train_df)]
            test_arr = np.c_[pre_processed_test_df, np.array(target_feature_test_df)]
            
            # This function saves the pre_processor_obj and it's path
            save_object(
                file_path = self.data_transformation_config.pre_processor_obj_file_path,
                obj = pre_processor_obj
            )

            # Return processed train and test array, along with pre processor object path
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.pre_processor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)
        