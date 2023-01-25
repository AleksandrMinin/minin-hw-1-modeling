import os
import pandas as pd

from train_test_split import tag_to_mlb, split_and_save_datasets
from src.config import TRAIN_DF, VALID_DF, TEST_DF

def remove_file(file_path):
    assert os.path.exists(file_path)
    os.remove(file_path)
    assert not os.path.exists(file_path)

def test_data_split():
    save_path = 'tests/data_for_tests/test_dataset'
    df = pd.read_csv(os.path.join(save_path, "train.csv"))
    df = tag_to_mlb(df)
    split_and_save_datasets(df, save_path)

    train_df = os.path.join(save_path, TRAIN_DF)
    train_amount = sum(1 for line in open(train_df, 'r')) - 1


    test_df = os.path.join(save_path, TEST_DF)
    test_amount = sum(1 for line in open(test_df, 'r')) - 1


    valid_df = os.path.join(save_path, VALID_DF)
    valid_amount = sum(1 for line in open(valid_df, 'r')) - 1

    assert train_amount + test_amount + valid_amount == 100

    remove_file(train_df)
    remove_file(test_df)
    remove_file(valid_df)
