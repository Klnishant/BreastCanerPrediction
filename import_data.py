import numpy as np
import sys
import pandas as pd
from sklearn.datasets import load_breast_cancer
from pymongo import MongoClient
from src.logger import logging
from src.exception import CustomException

try:
    # load the datset from sklearn
    data=load_breast_cancer()
    logging.info('data loaded succesfully')
    X=data.data
    y=data.target
    feature_names=data.feature_names

    #convert the data in to dataframe
    df=pd.DataFrame(X,columns=feature_names)
    df['Target']=y

    logging.info('Dataframe created')

    # convert the data frame in to list of dictionaries for insertion
    data_to_insert=df.to_dict(orient='records')
    logging.info('data frame converted in to list of dictionary')

    # mongo db atlas url
    mongo_uri='mongodb+srv://nishantkaushal:nishantkaushal@cluster0.vypsxre.mongodb.net/breast_cancer_assignment'

    #connect to mongodb atlas
    client=MongoClient(mongo_uri)
    db=client['breast_cancer_assignment']
    collection=db['breast_cancer_datasets']

    collection.insert_many(data_to_insert)

    client.close()
    logging.info('data inserted succesfully')
except Exception as e:
    raise CustomException(e,sys)