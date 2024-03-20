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
            "train-local-rule-length": train_results["rule_length"],
            "train-local-coverage": train_results["coverage"],
            "train-local-precision": train_results["precision"],
            "train-local-f1": train_results["f1"],
            "train-local-num-samples": train_results["num_samples"],
            "train-local-time-taken": train_results["time_taken"],
            "val-local-rule-length": val_results["rule_length"],
            "val-local-coverage": val_results["coverage"],
            "val-local-precision": val_results["precision"],
            "val-local-f1": val_results["f1"],
            "val-local-num-samples": val_results["num_samples"],
            "val-local-time-taken": val_results["time_taken"],
            "test-local-rule-length": test_results["rule_length"],
            "test-local-coverage": test_results["coverage"],
            "test-local-precision": test_results["precision"],
            "test-local-f1": test_results["f1"],
            "test-local-num-samples": test_results["num_samples"],
            "test-local-time-taken": test_results["time_taken"],
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
    wandb.log({
        "train-global-rule-length": global_train_results["global_rule_length"],
        "train-global-rule-coverage": global_train_results["global_rule_coverage"],
        "train-global-rule-precision": global_train_results["global_rule_precision"],
        "train-global-rule-f1": global_train_results["global_rule_f1"],
        "train-average-valid-rules": global_train_results["average_valid_rules"],
        "train-rule-length": global_train_results["rule_length"],
        "train-accuracy": global_train_results["accuracy"],
        "val-global-rule-length": global_val_results["global_rule_length"],
        "val-global-rule-coverage": global_val_results["global_rule_coverage"],
        "val-global-rule-precision": global_val_results["global_rule_precision"],
        "val-global-rule-f1": global_val_results["global_rule_f1"],
        "val-average-valid-rules": global_val_results["average_valid_rules"],
        "val-rule-length": global_val_results["rule_length"],
        "val-accuracy": global_val_results["accuracy"],
        "test-global-rule-length": global_test_results["global_rule_length"],
        "test-global-rule-coverage": global_test_results["global_rule_coverage"],
        "test-global-rule-precision": global_test_results["global_rule_precision"],
        "test-global-rule-f1": global_test_results["global_rule_f1"],
        "test-average-valid-rules": global_test_results["average_valid_rules"],
        "test-rule-length": global_test_results["rule_length"],
        "test-accuracy": global_test_results["accuracy"],
    })
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
