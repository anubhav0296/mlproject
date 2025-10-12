import os
import sys

from catboost import CatBoostRegressor
from sklearn.metrics import r2_score
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.logger import logging
from src.exception import CustomException

import pandas as pd
import numpy as np

from dataclasses import dataclass
from src.utils import evaluate_models, save_object

@dataclass
class ModelTrainerConfig:
    model_trainer_path = os.path.join("artifacts", "model_trainer.pxl")

class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            X_train, y_train, X_test, y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            # Dict of models where all the models are stored
            models = {
                    "Random Forest": RandomForestRegressor(),
                    "Decision Tree": DecisionTreeRegressor(),
                    "Gradient Boosting": GradientBoostingRegressor(),
                    "Linear Regression": LinearRegression(),
                    "XGBRegressor": XGBRegressor(),
                    "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                    "AdaBoost Regressor": AdaBoostRegressor(),
            }

            model_eval = evaluate_models(X_train, y_train, X_test, y_test, models)

            # To get the best model score from dictionary
            best_model_score = max(sorted(model_eval.values()))

            ## To get best model name from dict

            best_model_name = list(model_eval.keys())[
                list(model_eval.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No model is good enough")

            save_object(
                file_path = self.model_trainer_config.model_trainer_path,
                obj = best_model
            )

            predicted=best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)