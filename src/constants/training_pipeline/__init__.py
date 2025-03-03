import os
import sys

"""

Defining constants for training pipeline

"""
TARGET_COLUMN: str = "math_score"
PIPELINE_NAME: str = "StudentPlacement"
ARTIFACTS_DIR: str = "artfiacts"
FILE_NAME: str = "stud.csv"
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"


"""
Data Ingestion related constant start with DATA_INGESTION VAR Name

"""

DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATION: str = 0.2
