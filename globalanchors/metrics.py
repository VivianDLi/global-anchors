"""Calculating overall metrics given a dataset."""

from typing import List
from timeit import default_timer as timer

from tqdm import tqdm
import wandb

from globalanchors.combined.base import GlobalAnchors
from globalanchors.local.anchors import TextAnchors
from globalanchors.types import GlobalMetrics, LocalMetrics, Model


def calculate_local_metrics(
    explainer: TextAnchors,
    dataset: List[str],
    model: Model,
    log_to_wandb: bool = True,
) -> LocalMetrics:
    """Calculates local metrics for a given dataset.

    Args:
        explainer (TextAnchors): The local explainer.
        dataset (List[str]): The dataset to calculate metrics for.

    Returns:
        LocalMetrics: The calculated local metrics.
    """
    rule_lengths = []
    coverages = []
    precisions = []
    f1_scores = []
    num_samples = []
    times = []
    for data in tqdm(dataset):
        # time explanation
        start = timer()
        explanation = explainer.explain(data, model)
        end = timer()
        # log results
        times.append(end - start)
        rule_lengths.append(len(explanation["explanation"]))
        coverages.append(explanation["coverage"])
        precisions.append(explanation["precision"])
        num_samples.append(explanation["num_samples"])
        # check that precision and coverage are valid for f1
        assert (
            explanation["precision"] >= 0 and explanation["precision"] <= 1
        ), f"Precision must be between 0 and 1. Precision for example {explanation['example']} was {explanation['precision']} instead."
        assert (
            explanation["coverage"] >= 0 and explanation["coverage"] <= 1
        ), f"Coverage must be between 0 and 1. Coverage for example {explanation['example']} was {explanation['coverage']} instead."
        f1_scores.append(
            2
            * (explanation["precision"] * explanation["coverage"])
            / (explanation["precision"] + explanation["coverage"])
        )

        # log intermediate results to wandb
        if log_to_wandb:
            wandb.log(
                {
                    "local/intermediate/example": data,
                    "local/intermediate/rule-length": rule_lengths[-1],
                    "local/intermediate/coverage": coverages[-1],
                    "local/intermediate/precision": precisions[-1],
                    "local/intermediate/f1": f1_scores[-1],
                    "local/intermediate/num-samples": num_samples[-1],
                    "local/intermediate/time-taken": times[-1],
                }
            )

    return {
        "rule_length": sum(rule_lengths) / len(rule_lengths),
        "coverage": sum(coverages) / len(coverages),
        "precision": sum(precisions) / len(precisions),
        "f1": sum(f1_scores) / len(f1_scores),
        "num_samples": sum(num_samples) / len(num_samples),
        "time_taken": sum(times) / len(times),
    }


def calculate_global_metrics(
    explainer: GlobalAnchors, dataset: List[str], log_to_wandb: bool = True
) -> GlobalMetrics:
    """Calculates global metrics for a given dataset.

    Args:
        explainer (GlobalAnchors): The global explainer.
        dataset (List[str]): The dataset to calculate metrics for.

    Returns:
        GlobalMetrics: The calculated global metrics.
    """
    # calculate global ruleset metrics
    rule_lengths = []
    coverages = []
    precisions = []
    f1_scores = []
    for explanation in explainer.rules:
        rule_lengths.append(len(explanation["explanation"]))
        coverages.append(explanation["coverage"])
        precisions.append(explanation["precision"])
        # check that precision and coverage are valid for f1
        assert (
            explanation["precision"] >= 0 and explanation["precision"] <= 1
        ), f"Precision must be between 0 and 1. Precision for example {explanation['example']} was {explanation['precision']} instead."
        assert (
            explanation["coverage"] >= 0 and explanation["coverage"] <= 1
        ), f"Coverage must be between 0 and 1. Coverage for example {explanation['example']} was {explanation['coverage']} instead."
        f1_scores.append(
            2
            * (explanation["precision"] * explanation["coverage"])
            / (explanation["precision"] + explanation["coverage"])
        )
    global_rule_length = sum(rule_lengths) / len(rule_lengths)
    global_rule_coverage = sum(coverages) / len(coverages)
    global_rule_precision = sum(precisions) / len(precisions)
    global_rule_f1 = sum(f1_scores) / len(f1_scores)
    # calculate global prediction metrics
    num_rules = []
    rule_lengths = []
    accuracies = []
    for data in tqdm(dataset):
        output = explainer.explain(data)
        num_rules.append(len(output["explanations"]))
        if output["rule_used"] is not None:
            rule_lengths.append(len(output["rule_used"]["explanation"]))
        accuracies.append(
            1 if output["prediction"] == explainer.model([data])[0] else 0
        )
        # log intermediate results to wandb
        if log_to_wandb:
            wandb.log(
                {
                    "global/intermediate/example": data,
                    "global/intermediate/num-valid-rules": num_rules[-1],
                }
            )

    return {
        "global_rule_length": global_rule_length,
        "global_rule_coverage": global_rule_coverage,
        "global_rule_precision": global_rule_precision,
        "global_rule_f1": global_rule_f1,
        "average_valid_rules": sum(num_rules) / len(num_rules),
        "rule_length": sum(rule_lengths) / len(rule_lengths),
        "accuracy": sum(accuracies) / len(accuracies),
    }
