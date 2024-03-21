"""Main module to train, explain, and test a model. This should be the program entry point."""

import hydra
from sklearn.base import accuracy_score
import wandb
import numpy as np
from loguru import logger
from omegaconf import DictConfig

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
    # set seed for random number generators in numpy
    np.random.seed(cfg.seed)

    logger.info(f"Instantiating dataset: <{cfg.dataset._target_}...")
    dataloader: DataLoader = hydra.utils.instantiate(cfg.dataloader)

    logger.info(f"Instantiating model: <{cfg.model._target_}>...")
    model: BaseModel = hydra.utils.instantiate(cfg.model)

    logger.info(
        f"Instantiating and setting sampler for model: <{cfg.sampler._target_}>..."
    )
    sampler: NeighbourhoodSampler = hydra.utils.instantiate(cfg.sampler)
    model.set_sampler(sampler)

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
        name=cfg.name,
    )

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
            "model-val-accuracy": val_accuracy,
            "model-test-accuracy": test_accuracy,
        }
    )

    logger.info("Generating Local Explanations!")
    train_results = calculate_local_metrics(
        local_explainer, dataset["train_data"], model
    )
    val_results = calculate_local_metrics(
        local_explainer, dataset["val_data"], model
    )
    test_results = calculate_local_metrics(
        local_explainer, dataset["test_data"], model
    )
    wandb.log(
        {
            "local/train/rule-length": train_results["rule_length"],
            "local/train/coverage": train_results["coverage"],
            "local/train/precision": train_results["precision"],
            "local/train/f1": train_results["f1"],
            "local/train/num-samples": train_results["num_samples"],
            "local/train/time-taken": train_results["time_taken"],
            "local/val/rule-length": val_results["rule_length"],
            "local/val/coverage": val_results["coverage"],
            "local/val/precision": val_results["precision"],
            "local/val/f1": val_results["f1"],
            "local/val/num-samples": val_results["num_samples"],
            "local/val/time-taken": val_results["time_taken"],
            "local/test/rule-length": test_results["rule_length"],
            "local/test/coverage": test_results["coverage"],
            "local/test/precision": test_results["precision"],
            "local/test/f1": test_results["f1"],
            "local/test/num-samples": test_results["num_samples"],
            "local/test/time-taken": test_results["time_taken"],
        }
    )

    logger.info("Generating and Testing Global Explanations!")
    logger.info("Training on train data...")
    global_explainer.train(local_explainer, dataset["train_data"], model)
    logger.info("Testing on test data...")
    global_train_results = calculate_global_metrics(
        global_explainer, dataset["test_data"]
    )
    logger.info("Training on val data...")
    global_explainer.train(local_explainer, dataset["val_data"], model)
    logger.info("Testing on test data...")
    global_val_results = calculate_global_metrics(
        global_explainer, dataset["test_data"]
    )
    logger.info("Training on test data...")
    global_explainer.train(local_explainer, dataset["test_data"], model)
    logger.info("Testing on test data...")
    global_test_results = calculate_global_metrics(
        global_explainer, dataset["test_data"]
    )
    wandb.log(
        {
            "global/train/global-rule-length": global_train_results[
                "global_rule_length"
            ],
            "global/train/global-rule-coverage": global_train_results[
                "global_rule_coverage"
            ],
            "global/train/global-rule-precision": global_train_results[
                "global_rule_precision"
            ],
            "global/train/global-rule-f1": global_train_results[
                "global_rule_f1"
            ],
            "global/train/average-valid-rules": global_train_results[
                "average_valid_rules"
            ],
            "global/train/rule-length": global_train_results["rule_length"],
            "global/train/accuracy": global_train_results["accuracy"],
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
            "global/test/global-rule-length": global_test_results[
                "global_rule_length"
            ],
            "global/test/global-rule-coverage": global_test_results[
                "global_rule_coverage"
            ],
            "global/test/global-rule-precision": global_test_results[
                "global_rule_precision"
            ],
            "global/test/global-rule-f1": global_test_results[
                "global_rule_f1"
            ],
            "global/test/average-valid-rules": global_test_results[
                "average_valid_rules"
            ],
            "global/test/rule-length": global_test_results["rule_length"],
            "global/test/accuracy": global_test_results["accuracy"],
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
