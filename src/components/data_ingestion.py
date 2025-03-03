from src.exception.exception import CustomException
from src.logging.logger import logging
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation, DataTransformationConfig

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "raw.csv")

class DataIngestion:
    def __init__(self):
        self.dataIngestionConfig = DataIngestionConfig()
    
    def initiateDataIngestion(self):
        logging.info("Entered into Data Ingestion Component")

        try:
            df = pd.read_csv("notebook\data\stud.csv")
            logging.info("Read the dataset as dataframe")

            os.makedirs(os.path.dirname(self.dataIngestionConfig.train_data_path), exist_ok=True)

            df.to_csv(self.dataIngestionConfig.raw_data_path, header=True, index=False)

            logging.info("Splitting initiated")
            train_set, test_set = train_test_split(df, test_size=0.20, random_state=42)

            train_set.to_csv(self.dataIngestionConfig.train_data_path, header=True, index=False)
            test_set.to_csv(self.dataIngestionConfig.test_data_path, header=True, index=False)

            logging.info("Data Ingestion Completed")

            return (
                self.dataIngestionConfig.train_data_path,
                self.dataIngestionConfig.test_data_path
            )

        except Exception as e:
            raise CustomExcpetion(e,sys)
        
if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiateDataIngestion()

    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data, test_data)
        

