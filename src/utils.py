import os
import sys
from dataclasses import dataclass

import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException
from src.logger import logging

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

import numpy as np
import pandas as pd


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

# This function takes the inputs, runs a loop to fit the data in each model 
# the returns as reportof each model (Dictionary)   
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    
    try:
        report = {}

        for model_name, model in models.items():

            logging.info("Model Training has started")
            print(f"Training {model_name}...")

            # Get parameters for a particular model (using name of model - key)
            para = param.get(model_name, {})
            
            if para:
                # Perform GridSearchCV
                gs = GridSearchCV(model, para, cv=3)
                gs.fit(X_train, y_train)

                # Parameters that gave the best results on the hold out data.
                model.set_params(**gs.best_params_)

            # Fit the final parameters into the model
            model.fit(X_train, y_train)

            # predict and calculate r2
            y_pred_test = model.predict(X_test)
            test_model_r2_score = r2_score(y_test, y_pred_test)

            y_pred_train = model.predict(X_train)
            train_model_r2_score = r2_score(y_train, y_pred_train)

            logging.info(f"{model_name} has r2_score as {test_model_r2_score}")
            # save test score in report
            report[model_name] = test_model_r2_score

        return report

        
    except Exception as e:
        raise CustomException(e, sys)