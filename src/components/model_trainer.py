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
    # After training the model, the output file will be saved in below location
    model_trainer_path = os.path.join("artifacts", "model_trainer.pxl")

class ModelTrainer:
    # This function initializes the above Class and saves in a variable to make an object
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    # This method takes Input - train and test array after data transformation, then fits 
    # it to different models. At last it finds the r2_score of each model and the best 
    # performing model
    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Model training has initiated")

            # Split train and test array
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

            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }

            # This function takes the inputs, runs a loop to fit the data in each model 
            # the returns as reportof each model (Dictionary) 
            model_eval = evaluate_models(X_train, y_train, X_test, y_test, models, params)

            # To get the best model score from report dictionary
            best_model_score = max(sorted(model_eval.values()))

            ## To get best model name from dict
            best_model_name = list(model_eval.keys())[
                list(model_eval.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            # If the best models score is less than 60%, then throw an exception
            if best_model_score < 0.6:
                raise CustomException("No model is good enough")

            # This function saves the model object and it's path
            save_object(
                file_path = self.model_trainer_config.model_trainer_path,
                obj = best_model
            )
            
            # Find the r2_score of the best model
            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            logging.info(f"The best model is {best_model_name} with r2_score - {best_model_score}")

            return r2_square
                                            
        except Exception as e:
            raise CustomException(e, sys)