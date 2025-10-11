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
    pre_processor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    # This function helps in assigning an object to DataTransformationConfig
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig

    ''' This function is responsible for getting preprocessor object that does all the data transformation'''
    def get_data_transformation_object(self):
        try:
            logging.info("Getting data transformation object started")

            num_cols = ["reading_score","writing_score"]

            cat_cols = ["gender","race_ethnicity","parental_level_of_education","lunch","test_preparation_course"]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler(with_mean=True))
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder()),
                    ("scaler", StandardScaler())
                ]
            )

            logging.info("Numerical columns transformation is done")
            logging.info("Categorical column tarnsformation is done")

            pre_processor = ColumnTransformer(
                [
                ("num_pipeline", num_pipeline, num_cols),
                ("cat_pipeline", cat_pipeline, cat_cols)
            ]
            )

            logging.info("Preprocessing is completed")

            return pre_processor
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            target_col = ["math_score"]

            input_feature_train_df = train_df.drop(target_col, axis=1)
            target_feature_train_df = train_df[target_col]

            input_feature_test_df = test_df.drop(target_col, axis=1)
            target_feature_test_df = test_df[target_col]

            pre_processor_obj = self.get_data_transformation_object()

            pre_processed_train_df = pre_processor_obj.fit_transform(input_feature_train_df)
            pre_processed_test_df = pre_processor_obj.transform(input_feature_test_df)

            train_arr = np.c_[pre_processed_train_df, target_feature_train_df]
            test_arr = np.c_[pre_processed_test_df, target_feature_test_df]

            save_object(
                file_path = self.data_transformation_config.pre_processor_obj_file_path,
                obj = pre_processor_obj
            )


            return(
                train_arr,
                test_arr,
                self.data_transformation_config.pre_processor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)
        