import json
import os
import typing as tp
from dataclasses import asdict, dataclass, field

import albumentations as albu
import torch
from torch.optim.optimizer import Optimizer


@dataclass
class Config:
    num_workers: int
    seed: int
    loss: torch.nn.Module
    device: str
    optimizer: type(Optimizer)
    optimizer_kwargs: tp.Mapping
    scheduler: tp.Any
    scheduler_kwargs: tp.Mapping
    preprocessing: tp.Callable
    img_size: int
    augmentations: albu.Compose
    batch_size: int
    n_epochs: int
    early_stop_patience: int
    experiment_name: str
    model_kwargs: tp.Mapping
    log_metrics: tp.List[str]
    binary_thresh: float
    valid_metric: str
    minimize_metric: bool
    images_dir: str
    train_dataset_path: str
    valid_dataset_path: str
    test_dataset_path: str
    project_name: str
    checkpoints_dir: str = field(init=False)

    def to_dict(self) -> dict:
        res = {}
        for k, v in asdict(self).items():
            try:
                if isinstance(v, torch.nn.Module):
                    res[k] = v.__class__.__name__
                elif isinstance(v, dict):
                    res[k] = json.dumps(v, indent=4)
                else:
                    res[k] = str(v)
            except Exception:
                res[k] = str(v)
        return res

    def __post_init__(self):
        self.checkpoints_dir = os.path.join("./weights", self.experiment_name)  # noqa: WPS601
