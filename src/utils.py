import os
import sys
import pandas as pd
import numpy as np
import pickle
from src.logger import logging
from src.exception import CustomException

from sklearn.metrics import roc_auc_score


def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)
    except Exception as e:
        logging.info('Error occured in saving preprocessing object')
        raise CustomException(e,sys)
    
def model_evaluation(X_train,X_test,y_train,y_test,models):
    try:
        report={}

        for i in range(len(models)):
            model=list(models.values())[i]
            model.fit(X_train,y_train)

            y_test_pred=model.predict(X_test)

            test_model_score=roc_auc_score(y_test,y_test_pred)

            report[list(models.keys())[i]] = test_model_score
        return report
    except Exception as e:
        logging.info('Error occured in model training')
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Error occured in load object utils')
        raise CustomException(e,sys)