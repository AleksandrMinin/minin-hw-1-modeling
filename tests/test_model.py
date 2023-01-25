import os
import cv2
import numpy as np
import torch
import timm

from src.config import config
from src.config import IMG_SIZE
from src.dataset import get_class_names, get_loaders
from src.tools import preprocess_imagenet


TESTS_DIR = os.path.dirname(__file__)

def test_forward():
    loaders, infer_loader = get_loaders(config)
    class_names = get_class_names(config)

    model = timm.create_model(num_classes=len(class_names), **config.model_kwargs)
    image = cv2.imread(os.path.join(TESTS_DIR, 'data_for_tests/train-jpg', 'train_0.jpg'))
    image = image.astype(np.float32)
    image = preprocess_imagenet(image, img_size=IMG_SIZE)
    model(torch.from_numpy(image)[None]).shape == (1, len(class_names))
    assert model(torch.from_numpy(image)[None]).shape == (1, len(class_names))
