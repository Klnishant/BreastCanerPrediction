import sys
import os
from src.logger import logging
from src.exception import CustomException
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
import pandas as pd
from dataclasses import dataclass

@dataclass
class DataIngestionConfig():
    train_data_path=os.path.join('artifacts','train_data.csv')
    test_data_path=os.path.join('artifacts','test_data.csv')
    row_data_path=os.path.join('artifacts','row_data.csv')

class DataIngestion():
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        try:
            logging.info('Data ingestion starts')

            # mongo db atlas url
            mongo_uri='mongodb+srv://nishantkaushal:nishantkaushal@cluster0.vypsxre.mongodb.net/breast_cancer_assignment'

            #connect to mongodb atlas
            client=MongoClient(mongo_uri)
            db=client['breast_cancer_assignment']
            collection=db['breast_cancer_datasets']
            logging.info('Connection to mongodb succesfull')

            # retrive the data from mongodb
            data_list=list(collection.find())
            logging.info('Retrive data from mongodb success')

            #close the mongodb connection
            client.close()

            # convert the data in to dataframe
            df=pd.DataFrame(data_list)


            #remove _id column which is automatically add by mongodb
            df=df.drop(columns='_id',axis=1)
            logging.info(f'row data head:{df.head()}')

            os.makedirs(os.path.dirname(self.ingestion_config.row_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.row_data_path,index=False)

            train_data,test_data=train_test_split(df,test_size=0.25,random_state=42)

            train_data.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_data.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info('data ingestion is complete')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)

    