import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from sklearn.metrics import r2_score

from src.exception.exception import CustomException
from src.logging.logger import logging

from src.utils.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Splitting train and test data")

            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],

            )

            models = {
                "CatBoostRegressor":CatBoostRegressor(),
                "AdaBoostRegressor":AdaBoostRegressor(),
                "GradientBoostingRegressor":GradientBoostingRegressor(),
                "RandomForestRegressor":RandomForestRegressor(),
                "LinearRegression":LinearRegression(),
                "KNeighborsRegressor":KNeighborsRegressor(),
                "DecisionTreeRegressor":DecisionTreeRegressor(),
                "XGBRegressor":XGBRegressor()
            }


            model_report:dict  = evaluate_models(X_train=X_train, y_train=y_train, 
                                            X_test=X_test,y_test=y_test, models=models)


            ## To get the best model scroe from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get the best model name from dict
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("Best model not found")
            
            logging.info("Best model found on both train and test data")

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            score = r2_score(y_test, predicted)
            
            return score

        except Exception as e:
            raise CustomException(e,sys)

