import logging
import os

import pandas as pd
from src.dataset_splitter import stratify_shuffle_split_subsets
from src.config import ROOT_PATH, IMG_COLUMN, LABEL_COLUMN, TRAIN_DF, VALID_DF, TEST_DF
from sklearn.preprocessing import MultiLabelBinarizer


def tag_to_mlb(df: pd.DataFrame):
    mlb = MultiLabelBinarizer()
    df_mlb = mlb.fit_transform(df.pop(LABEL_COLUMN).str.split())
    df_mlb = pd.DataFrame(df_mlb, index=df.index, columns=mlb.classes_)
    df = df.join(df_mlb)
    return df
    

def split_and_save_datasets(df: pd.DataFrame, save_path: str):
    logging.info(f"Original dataset: {len(df)}")
    df = df.drop_duplicates()
    logging.info(f"Final dataset: {len(df)}")
    
    train_df, valid_df, test_df = stratify_shuffle_split_subsets(
        df,
        img_path_column=IMG_COLUMN,
        train_fraction=0.8,
        verbose=True,
    )
    logging.info(f"Train dataset: {len(train_df)}")
    logging.info(f"Valid dataset: {len(valid_df)}")
    logging.info(f"Test dataset: {len(test_df)}")

    train_df.to_csv(os.path.join(save_path, TRAIN_DF), index=False)
    valid_df.to_csv(os.path.join(save_path, VALID_DF), index=False)
    test_df.to_csv(os.path.join(save_path, TEST_DF), index=False)
    logging.info("Datasets successfully saved!")


if __name__ == "__main__":
    save_path = ROOT_PATH
    logging.basicConfig(level=logging.INFO)
    df = pd.read_csv(os.path.join(ROOT_PATH, "train.csv"))
    df = tag_to_mlb(df)
    split_and_save_datasets(df, save_path)
