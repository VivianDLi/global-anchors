import os

import omegaconf
from hydra.utils import instantiate

from globalanchors import constants
from globalanchors.datasets import DataLoader

DATALOADER_CONFIG_DIR = constants.HYDRA_CONFIG_PATH / "dataloader"


def test_instantiate_datasets():
    """Test we can instantiate all datasets."""
    for t in os.listdir(DATALOADER_CONFIG_DIR):
        config_path = DATALOADER_CONFIG_DIR / t
        cfg = omegaconf.OmegaConf.load(config_path)

        dataset = instantiate(cfg)

        assert dataset, f"Dataset {t} not instantiated!"
        assert isinstance(dataset, DataLoader)


def test_datasets_have_data_attr():
    """Test dataloaders have a correct dataset attribute."""
    for t in os.listdir(DATALOADER_CONFIG_DIR):
        config_path = DATALOADER_CONFIG_DIR / t
        cfg = omegaconf.OmegaConf.load(config_path)

        dl = instantiate(cfg)
        assert hasattr(
            dl, "dataset"
        ), f"Datamodules {dl} has no overwrite attribute"
        dataset = dl.dataset
        current_keys = set(dataset.keys())
        expected_keys = set(
            [
                "train_data",
                "train_labels",
                "val_data",
                "val_labels",
                "test_data",
                "test_labels",
            ]
        )
        assert (
            current_keys == expected_keys
        ), f"Dataset for cfg {cfg} is missing keys {expected_keys - current_keys}."
