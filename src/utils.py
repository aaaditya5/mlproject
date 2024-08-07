import os
import sys
import numpy as np 
import pandas as pd 
import dill
from sklearn.metrics import r2_score,mean_squared_error
from src.exception import CustomException
def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    

from sklearn.metrics import r2_score

def evaluate_model(X_train, y_train, X_test, y_test, models):
    """
    Evaluates a dictionary of models and returns a report with test R-squared scores.

    Args:
        X_train (array-like): Training features.
        y_train (array-like): Training target labels.
        X_test (array-like): Testing features.
        y_test (array-like): Testing target labels.
        models (dict): Dictionary of models to evaluate, where keys are model names
                      and values are the models themselves.

    Returns:
        dict: A dictionary containing the test R-squared scores for each model.
    """

    try:
        report = {}
        for model_name, model in models.items():
            model.fit(X_train, y_train)

            # Use a robust metric like mean squared error (MSE) for training evaluation
            y_train_pred = model.predict(X_train)
            train_model_score = mean_squared_error(y_train, y_train_pred)  # Avoid R-squared for training

            y_test_pred = model.predict(X_test)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)  # Assuming CustomException is defined elsewhere
    


#it is loading the pickel file 
def load_object(file_path):
    try:
        with open (file_path,"rb") as file_obj:
            return dill.load(file_obj)
        
    except Exception as e:
        raise CustomException(e,sys)