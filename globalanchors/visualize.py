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
    
    logger.info("Only run on an example of the data.")
    data = dataset["test_data"][50]
    
    fitness = calculate_genetic_metrics(sampler, model, [data])["fitness"]
    wandb.log({"fitness": np.mean(fitness), "fitnesses": fitness})


def compare_examples():
    import omegaconf
    from globalanchors.local.neighbourhood.genetic import (
        GeneticAlgorithmSampler,
    )
    from globalanchors.local.neighbourhood.pos import PartOfSpeechSampler
    from globalanchors.local.neighbourhood.unk import UnkTokenSampler
    from globalanchors.anchor_types import (
        CandidateAnchor,
        InputData,
        BeamState,
    )

    sample_text = "This is a good book."
    anchor = CandidateAnchor(feats=set(["good"]), feat_indices=set([3]))

    model_config = constants.HYDRA_CONFIG_PATH / "model" / "svm.yaml"
    dataloader_config = (
        constants.HYDRA_CONFIG_PATH / "dataloader" / "polarity.yaml"
    )

    model = hydra.utils.instantiate(omegaconf.OmegaConf.load(model_config))
    dataloader = hydra.utils.instantiate(
        omegaconf.OmegaConf.load(dataloader_config)
    )
    model.train(dataloader.dataset)

    genetic_sampler = GeneticAlgorithmSampler(
        mutation_prob=0.2, n_generations=10
    )
    pos_sampler = PartOfSpeechSampler()
    unk_sampler = UnkTokenSampler()

    example = InputData(sample_text, model)
    neighbourhood = unk_sampler.sample(
        example,
        model,
        n=1,
    )
    neighbourhood.reallocate()
    state = BeamState(
        neighbourhood,
        {},
        np.ones((10, len(example.tokens))),
        len(example.tokens),
        example,
        model,
    )
    state.initialize_features()

    genetic_samples_no_anchor = genetic_sampler.sample(
        example, model, n=10
    ).string_data
    logger.info(f"Genetic samples without anchor: {genetic_samples_no_anchor}")
    pos_samples_no_anchor = pos_sampler.sample(
        example, model, n=10
    ).string_data
    logger.info(f"POS samples without anchor: {pos_samples_no_anchor}")
    unk_samples_no_anchor = unk_sampler.sample(
        example, model, n=10
    ).string_data
    logger.info(f"UNK samples without anchor: {unk_samples_no_anchor}")

    genetic_samples_with_anchor = genetic_sampler.sample_candidate_with_state(
        anchor, state, n=10
    )[1].neighbourhood.string_data
    logger.info(
        f"Genetic samples with anchor: {[' '.join(row) for row in genetic_samples_with_anchor[1:11]]}"
    )
    pos_samples_with_anchor = pos_sampler.sample_candidate_with_state(
        anchor, state, n=10
    )[1].neighbourhood.string_data
    logger.info(
        f"POS samples with anchor: {[' '.join(row) for row in pos_samples_with_anchor[11:21]]}"
    )
    unk_samples_with_anchor = unk_sampler.sample_candidate_with_state(
        anchor, state, n=10
    )[1].neighbourhood.string_data
    logger.info(
        f"UNK samples with anchor: {[' '.join(row) for row in unk_samples_with_anchor[21:31]]}"
    )


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
    compare_examples()
    # run_experiment()
