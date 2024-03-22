"""Visualizing and tuning genetic algorithm parameters."""

import hydra
import wandb
import numpy as np
from omegaconf import DictConfig


# adding globalanchors to path
import pathlib, sys

sys.path.append(str(pathlib.Path(__file__).parent.parent))

from globalanchors import constants
from globalanchors.datasets import DataLoader
from globalanchors.local.neighbourhood.base import NeighbourhoodSampler
from globalanchors.models import BaseModel
from globalanchors.metrics import calculate_genetic_metrics

from loguru import logger

def genetic_algorithm_tuning(cfg: DictConfig):
    """Evaluates a genetic algorithm sampler on a config.

    The dataloader is instantiated from ``cfg.dataloader``.
    The model is instantiated from ``cfg.model``.
    The genetic algorithm sampler is instantiated from ``cfg.sampler``.

    Args:
        cfg (DictConfig): DictConfig containing the experiment configuration.
    """
    logger.info("Initializing wandb...")
    wandb.init(
        project=constants.WANDB_PROJECT,
        entity=constants.WANDB_ENTITY,
        name=cfg.name,
        group=cfg.group,
    )
    
    # set seed for random number generators in numpy
    np.random.seed(cfg.seed)

    logger.info(f"Instantiating dataset: <{cfg.dataloader._target_}...")
    dataloader: DataLoader = hydra.utils.instantiate(cfg.dataloader)

    logger.info(f"Instantiating model: <{cfg.model._target_}>...")
    model: BaseModel = hydra.utils.instantiate(cfg.model)

    logger.info(
        f"Instantiating and setting sampler for local explainer: <{cfg.sampler._target_}>..."
    )
    sampler: NeighbourhoodSampler = hydra.utils.instantiate(cfg.sampler)
    
    dataset = dataloader.dataset
    logger.info("Starting training!")
    model.train(dataset)
    
    fitness = calculate_genetic_metrics(sampler, model, dataset["test_data"])["fitness"]
    wandb.log({"fitness": np.mean(fitness), "fitnesses": fitness})
    
# Load hydra config from yaml files and command line arguments.
@hydra.main(
    version_base="1.3",
    config_path=str(constants.HYDRA_CONFIG_PATH),
    config_name="visualize",
)
def run_experiment(cfg: DictConfig) -> None:
    """Load the hydra config."""
    genetic_algorithm_tuning(cfg)


if __name__ == "__main__":
    run_experiment()
