"""Main module to train, explain, and test a model. This should be the program entry point."""

import hydra
import wandb
import numpy as np
from loguru import logger
from omegaconf import DictConfig

from globalanchors import constants
from globalanchors.combined.base import GlobalAnchors
from globalanchors.datasets import DataLoader
from globalanchors.local.anchors import TextAnchors
from globalanchors.models import BaseModel


def explain_model(cfg: DictConfig):
    """Trains and evaluates an explainer on a config.

    The dataloader is instantiated from ``cfg.dataloader``.
    The model is instantiated from ``cfg.model``.
    The local explainer is instantiated from ``cfg.local_explainer``.
    The global explainer is instantiated from ``cfg.global_explainer``.
    The model/explainer is always trained and tested and all available metrics are logged.

    Args:
        cfg (DictConfig): DictConfig containing the experiment configuration.
    """
    # set seed for random number generators in numpy
    np.random.seed(cfg.seed)

    logger.info(f"Instantiating dataset: <{cfg.dataset._target_}...")
    dataloader: DataLoader = hydra.utils.instantiate(cfg.dataloader)

    logger.info(f"Instantiating model: <{cfg.model._target_}>...")
    model: BaseModel = hydra.utils.instantiate(cfg.model)

    logger.info(
        f"Instantiating local explainer: <{cfg.local_explainer._target_}>..."
    )
    local_explainer: TextAnchors = hydra.utils.instantiate(cfg.local_explainer)

    logger.info(
        f"Instantiating global explainer: <{cfg.global_explainer._target_}>..."
    )
    global_explainer: GlobalAnchors = hydra.utils.instantiate(
        cfg.global_explainer
    )

    logger.info(f"Initializing Wandb...")
    wandb.init(
        project=constants.WANDB_PROJECT,
        entity=constants.WANDB_ENTITY,
        config=cfg,
        name="",
        group="",
    )

    logger.info("Starting training!")
    # TODO: log training results to wandb

    logger.info("Generating Local Explanations!")
    # TODO: log local explanations to wandb

    logger.info("Generating Global Explanations!")
    # TODO: log global explanations to wandb

    logger.info("Starting testing!")
    logger.info("Testing Local Explanations!")
    # TODO: testing local explanations and logging to wandb

    logger.info("Testing Global Explanations!")
    # TODO: testing global explanations and logging to wandb


# Load hydra config from yaml files and command line arguments.
@hydra.main(
    version_base="1.3",
    config_path=str(constants.HYDRA_CONFIG_PATH),
    config_name="explain",
)
def run_experiment(cfg: DictConfig) -> None:
    """Load the hydra config."""
    explain_model(cfg)


if __name__ == "__main__":
    run_experiment()
