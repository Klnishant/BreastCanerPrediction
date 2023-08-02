import pandas as pd
import numpy as np
import sys
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
from dataclasses import dataclass

@dataclass
class DataTransformationConfig():
    preprocessor_obj_file=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation():
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformation_obj(self):
        try:
            preprocessor=Pipeline(
                steps=[('imputer',SimpleImputer()),('scaler',StandardScaler())]
            )

            logging.info('Preprocessing object created')
            
            return preprocessor
        except Exception as e:
            logging.info('Error occured in creating preprocessing object')
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            logging.info('Data transformation initiated')
            #reading train and test data
            train_df= pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info('Reading of training and test data is completed')
            logging.info(f'Train dataframe head: \n{train_df.head().to_string()}')
            logging.info(f'Test data head: \n{test_df.head().to_string()}')

            preprocessing_obj=self.get_data_transformation_obj()
            logging.info('Preprocessing object obtained')

            target_col='Target'

            input_feature_train_df=train_df.drop(columns=target_col,axis=1)
            target_feature_train_df=train_df[target_col]

            input_feature_test_df=test_df.drop(columns=target_col,axis=1)
            target_feature_test_df=test_df[target_col]

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            logging.info('transformation applied')

            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file,
                obj=preprocessing_obj
            )

            logging.info('preprocessor obj file in and save')

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file,
            )
        except Exception as e:
            logging.info('Error occured in data transformation')
            raise CustomException(e,sys)