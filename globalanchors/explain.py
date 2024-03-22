"""Main module to train, explain, and test a model. This should be the program entry point."""

import hydra
from sklearn.metrics import accuracy_score
import wandb
import numpy as np
from omegaconf import DictConfig


# adding globalanchors to path
import pathlib, sys

sys.path.append(str(pathlib.Path(__file__).parent.parent))

from globalanchors import constants
from globalanchors.combined.base import GlobalAnchors
from globalanchors.datasets import DataLoader
from globalanchors.local.anchors import TextAnchors
from globalanchors.local.neighbourhood.base import NeighbourhoodSampler
from globalanchors.metrics import (
    calculate_local_metrics,
    calculate_global_metrics,
)
from globalanchors.models import BaseModel


from loguru import logger


def explain_model(cfg: DictConfig):
    """Trains and evaluates an explainer on a config.

    The dataloader is instantiated from ``cfg.dataloader``.
    The model is instantiated from ``cfg.model``.
    The local explainer is instantiated from ``cfg.local_explainer``.
    The neighbourhood sampling method used in the local explainer is instantiated from ``cfg.sampler``.
    The global explainer is instantiated from ``cfg.global_explainer``.
    The model/explainer is always trained and tested and all available metrics are logged.

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

    logger.info(f"Instantiating local explainer: <{cfg.local._target_}>...")
    local_explainer: TextAnchors = hydra.utils.instantiate(cfg.local)

    logger.info(
        f"Instantiating and setting sampler for local explainer: <{cfg.sampler._target_}>..."
    )
    sampler: NeighbourhoodSampler = hydra.utils.instantiate(cfg.sampler)
    local_explainer.set_functions(sampler, model)

    logger.info(
        f"Instantiating global explainer: <{cfg.combined._target_}>..."
    )
    global_explainer: GlobalAnchors = hydra.utils.instantiate(cfg.combined)

    dataset = dataloader.dataset
    logger.info("Starting training!")
    model.train(dataset)
    val_accuracy = accuracy_score(
        dataset["val_labels"], model(dataset["val_data"])
    )
    test_accuracy = accuracy_score(
        dataset["test_labels"], model(dataset["test_data"])
    )
    wandb.log(
        {
            "model/val/accuracy": val_accuracy,
            "model/test/accuracy": test_accuracy,
        }
    )

    logger.info("Getting small subset for compute...")
    data = dataset["val_data"][:50]

    logger.info("Generating Local Explanations!")
    local_results = calculate_local_metrics(local_explainer, data)
    wandb.log(
        {
            "local/rule-length": local_results["rule_length"],
            "local/coverage": local_results["coverage"],
            "local/precision": local_results["precision"],
            "local/f1": local_results["f1"],
            "local/num-samples": local_results["num_samples"],
            "local/time-taken": local_results["time_taken"],
        }
    )

    logger.info("Generating and Testing Global Explanations!")
    logger.info("Training on val data...")
    global_explainer.train(local_explainer, data)
    logger.info("Testing on test data...")
    global_val_results = calculate_global_metrics(
        global_explainer, dataset["test_data"]
    )
    wandb.log(
        {
            "global/val/global-rule-length": global_val_results[
                "global_rule_length"
            ],
            "global/val/global-rule-coverage": global_val_results[
                "global_rule_coverage"
            ],
            "global/val/global-rule-precision": global_val_results[
                "global_rule_precision"
            ],
            "global/val/global-rule-f1": global_val_results["global_rule_f1"],
            "global/val/average-valid-rules": global_val_results[
                "average_valid_rules"
            ],
            "global/val/rule-length": global_val_results["rule_length"],
            "global/val/accuracy": global_val_results["accuracy"],
            "global/val/coverage": global_val_results["coverage"],
        }
    )
    logger.info("Completed Tests!")


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
