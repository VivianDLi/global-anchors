"""Calculating overall metrics given a dataset."""

from typing import List
from timeit import default_timer as timer

import numpy as np
from loguru import logger
import wandb

from globalanchors.local.neighbourhood.genetic import GeneticAlgorithmSampler
from globalanchors.combined.base import GlobalAnchors
from globalanchors.local.anchors import TextAnchors
from globalanchors.anchor_types import (
    GeneticMetrics,
    GlobalMetrics,
    InputData,
    LocalMetrics,
    Model,
)


def calculate_local_metrics(
    explainer: TextAnchors,
    dataset: List[str],
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
    for i, data in enumerate(dataset):
        logger.info(f"Explaining example {data} ({i} / {len(dataset)})...")
        # time explanation
        start = timer()
        explanation = explainer.explain(data)
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
    covered = []
    for i, data in enumerate(dataset):
        logger.info(f"Explaining example {data} ({i} / {len(dataset)})...")
        output = explainer.explain(data)
        num_rules.append(len(output["explanations"]))
        if output["rule_used"] is not None:
            rule_lengths.append(len(output["rule_used"]["explanation"]))
        accuracies.append(
            1
            if output["prediction"] == explainer.explainer.model([data])[0]
            else 0
        )
        covered.append(1 if len(output["explanations"]) > 0 else 0)
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
        "rule_length": (
            sum(rule_lengths) / len(rule_lengths)
            if len(rule_lengths) > 0
            else 0
        ),
        "accuracy": sum(accuracies) / len(accuracies),
        "coverage": sum(covered) / len(covered),
    }


def calculate_genetic_metrics(
    sampler: GeneticAlgorithmSampler, model: Model, dataset: List[str]
) -> GeneticMetrics:
    fitnesses = []
    examples = wandb.Table(
        columns=[
            "Input",
            "Anchor",
            "PosEx. 1",
            "PosEx. 2",
            "PosEx. 3",
            "PosEx. 4",
            "PosEx. 5",
            "NegEx. 1",
            "NegEx. 2",
            "NegEx. 3",
            "NegEx. 4",
            "NegEx. 5",
        ]
    )
    for data in dataset:
        # initialize random anchor
        if type(data) == bytes:
            logger.debug("Decoding byte string example.")
            data = data.decode()
        assert (
            type(data) == str
        ), f"Input must be either a string or a byte array. Input was instead a {type(data)}."
        logger.info(f"Data: {data}")
        example = InputData(data, model)
        probabilities = np.ones((5, len(example.tokens))) * 0.2
        random_sample = np.random.uniform(0, 1, size=probabilities.shape)
        data_idx = (random_sample < probabilities).astype(int)
        # perturb 5 random samples
        required_features = example.tokens[
            data_idx[0] == 1
        ].tolist()  # keep anchor constant - set first row as fixed
        _, string_data, indv_fitnesses = sampler.generate_samples(
            example, 10, required_features, model
        )
        # add average population fitness
        population_fitness = sum(indv_fitnesses) / len(indv_fitnesses)
        fitnesses.append(population_fitness)
        # log examples
        table_data = (
            [example.text]
            + [f"({', '.join(required_features)})"]
            + string_data
        )
        examples.add_data(*table_data)
        wandb.log({"fitness": population_fitness, "examples": examples})

    return {"fitness": np.array(fitnesses)}
