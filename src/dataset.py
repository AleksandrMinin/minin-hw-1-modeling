import logging
import os
import typing as tp
from collections import OrderedDict
import albumentations as albu

import cv2
import numpy as np
import pandas as pd
from src.config import Config, IMG_COLUMN
from src.const import IMAGES, TARGETS
from src.tools import worker_init_fn
from torch.utils.data import DataLoader, Dataset


class AmazonDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        config: Config,
        augmentation: tp.Optional[albu.Compose] = None,
        preprocessing: tp.Optional[tp.Callable] = None,
    ):
        self.df = df
        self.config = config
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, idx: int) -> tp.Dict[str, np.ndarray]:
        row = self.df.iloc[idx]

        img_name = os.path.join(self.config.images_dir, row[IMG_COLUMN])
        img_path = f"{img_name}.jpg"
        target = np.array(row.drop([IMG_COLUMN]), dtype="float32")

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.augmentation:
            image = self.augmentation(image=image)["image"]
        if self.preprocessing:
            image = self.preprocessing(image)

        return {IMAGES: image, TARGETS: target}

    def __len__(self) -> int:
        return len(self.df)


def _get_dataframes(config: Config) -> tp.Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(config.train_dataset_path)
    valid_df = pd.read_csv(config.valid_dataset_path)
    test_df = pd.read_csv(config.test_dataset_path)

    logging.info(f"Train dataset: {len(train_df)}")  # noqa: WPS237
    logging.info(f"Valid dataset: {len(valid_df)}")  # noqa: WPS237
    logging.info(f"Test dataset: {len(test_df)}")    # noqa: WPS237

    return train_df, valid_df, test_df


def get_class_names(config: Config) -> tp.List[str]:
    class_names = list(pd.read_csv(config.train_dataset_path, nrows=1).drop(IMG_COLUMN, axis=1))
    logging.info(f"Classes num: {len(class_names)}")  # noqa: WPS237
    return class_names


def get_datasets(config: Config) -> tp.Tuple[Dataset, Dataset, Dataset]:  # noqa: WPS210
    df_train, df_val, df_test = _get_dataframes(config)

    train_dataset = AmazonDataset(
        df_train,
        config,
        augmentation=config.augmentations,
        preprocessing=config.preprocessing,
    )

    valid_dataset = AmazonDataset(
        df_val,
        config,
        preprocessing=config.preprocessing,
    )

    test_dataset = AmazonDataset(
        df_test,
        config,
        preprocessing=config.preprocessing,
    )

    return train_dataset, valid_dataset, test_dataset


def get_loaders(                                                           # noqa: WPS210
    config: Config,
) -> tp.Tuple[tp.OrderedDict[str, DataLoader], tp.Dict[str, DataLoader]]:  # noqa: WPS221
    train_dataset, valid_dataset, test_dataset = get_datasets(config)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        worker_init_fn=worker_init_fn,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    return OrderedDict({"train": train_loader, "valid": valid_loader}), {"infer": test_loader}
