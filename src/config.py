from datetime import datetime
from functools import partial
import os

import albumentations as albu
import torch
from src.base_config import Config
from src.tools import preprocess_imagenet
from torch.nn import BCEWithLogitsLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau


SEED = 42
IMG_SIZE = 256
BATCH_SIZE = 200
N_EPOCHS = 50
ROOT_PATH = os.path.join(os.environ.get("ROOT_PATH"))
IMAGES_DIR = "train-jpg"
IMG_COLUMN = "image_name"
LABEL_COLUMN = "tags"
TRAIN_DF = "train_df.csv"
VALID_DF = "valid_df.csv"
TEST_DF = "test_df.csv"

hue_shift_limit = 20
sat_shift_limit = 30
val_shift_limit = 20
p_hue = 0.5

brightness_limit = 0.2
contrast_limit = 0.2
p_contrast = 0.5

date_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")


augmentations = albu.Compose(
    [
        albu.HorizontalFlip(),
        albu.VerticalFlip(),
        albu.HueSaturationValue(
            hue_shift_limit=hue_shift_limit,
            sat_shift_limit=sat_shift_limit,
            val_shift_limit=val_shift_limit,
            p=p_hue,
        ),
        albu.RandomBrightnessContrast(
            brightness_limit=brightness_limit,
            contrast_limit=contrast_limit,
            p=p_contrast,
        ),
        albu.ShiftScaleRotate(),
        albu.GaussianBlur(),
    ],
)


config = Config(
    num_workers=4,
    seed=SEED,
    loss=BCEWithLogitsLoss(),
    device="cuda",
    optimizer=torch.optim.Adam,
    optimizer_kwargs={
        "lr": 1e-3,
        "weight_decay": 5e-4,
    },
    scheduler=ReduceLROnPlateau,
    scheduler_kwargs={
        "mode": "min",
        "factor": 0.1,
        "patience": 5,
    },
    img_size=IMG_SIZE,
    augmentations=augmentations,
    preprocessing=partial(preprocess_imagenet, img_size=IMG_SIZE),
    batch_size=BATCH_SIZE,
    n_epochs=N_EPOCHS,
    early_stop_patience=10,
    model_kwargs={"model_name": "resnet18", "pretrained": True},
    log_metrics=["auc", "f1"],
    binary_thresh=0.1,
    valid_metric="auc",
    minimize_metric=False,
    images_dir=os.path.join(ROOT_PATH, IMAGES_DIR),
    train_dataset_path=os.path.join(ROOT_PATH, TRAIN_DF),
    valid_dataset_path=os.path.join(ROOT_PATH, VALID_DF),
    test_dataset_path=os.path.join(ROOT_PATH, TEST_DF),
    project_name="ClassificationImagesFromSpace",
    experiment_name=f"experiment_3_{date_time}",
)
