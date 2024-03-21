import os

import numpy as np
import omegaconf
from hydra.utils import instantiate
from loguru import logger

from globalanchors import constants
from globalanchors.local.neighbourhood.base import NeighbourhoodSampler
from globalanchors.types import (
    BeamState,
    CandidateAnchor,
    InputData,
    NeighbourhoodData,
)

SAMPLER_CONFIG_DIR = constants.HYDRA_CONFIG_PATH / "sampler"


def test_instantiate_samplers():
    """Test we can instantiate all neighbourhood samplers."""
    for t in os.listdir(SAMPLER_CONFIG_DIR):
        config_path = SAMPLER_CONFIG_DIR / t
        cfg = omegaconf.OmegaConf.load(config_path)

        sampler = instantiate(cfg)

        assert sampler, f"Sampler {t} not instantiated!"
        assert isinstance(sampler, NeighbourhoodSampler)


def test_perturb_sample():
    """Test perturb samples function runs with correct results."""
    for t in os.listdir(SAMPLER_CONFIG_DIR):
        logger.info("Testing sampler: {t}")
        config_path = SAMPLER_CONFIG_DIR / t
        cfg = omegaconf.OmegaConf.load(config_path)

        sampler = instantiate(cfg)
        # initialize test data
        test_model = lambda x: [1 for _ in x]
        test_example = InputData(
            text="This is a test sentence.", model=test_model
        )
        test_data = np.array(
            [
                [0, 1, 0, 1, 0, 0],
                [0, 1, 0, 1, 0, 0],
                [0, 1, 0, 1, 0, 0],
            ]
        )
        expected_labels = np.array([1, 1, 1])
        # run test
        string_data, data, labels = sampler.perturb_samples(
            test_example, test_data, test_model, True
        )
        assert (
            len(string_data) == 3
        ), f"Expected 3 samples, got {len(string_data)}."
        assert data.shape == (
            3,
            6,
        ), f"Expected data shape (3, 6), got {data.shape}."
        assert all(
            [labels[i] == expected_labels[i] for i in range(len(labels))]
        ), f"Expected labels {expected_labels}, got {labels}."


def test_unbiased_sample():
    """Test unbiased sampling function runs."""
    for t in os.listdir(SAMPLER_CONFIG_DIR):
        logger.info("Testing sampler: {t}")
        config_path = SAMPLER_CONFIG_DIR / t
        cfg = omegaconf.OmegaConf.load(config_path)

        sampler = instantiate(cfg)
        # initialize test data
        test_model = lambda x: [1 for _ in x]
        test_example = InputData(
            text="This is a test sentence.", model=test_model
        )
        n = 5
        # run test
        neighbourhood = sampler.sample(test_example, test_model, n)
        assert isinstance(
            neighbourhood, NeighbourhoodData
        ), f"Expected NeighbourhoodData, got {type(neighbourhood)}."
        assert neighbourhood.string_data.shape == (
            n,
            1,
        ), f"Expected string data shape ({n}, 1), got {neighbourhood.string_data.shape}."
        assert neighbourhood.data.shape == (
            n,
            6,
        ), f"Expected data shape ({n}, 6), got {neighbourhood.data.shape}."
        assert neighbourhood.labels.shape == (
            n,
        ), f"Expected labels shape ({n},), got {neighbourhood.labels.shape}."


def test_sample_with_state():
    """Test sample with state function runs."""
    for t in os.listdir(SAMPLER_CONFIG_DIR):
        logger.info("Testing sampler: {t}")
        config_path = SAMPLER_CONFIG_DIR / t
        cfg = omegaconf.OmegaConf.load(config_path)

        sampler = instantiate(cfg)
        # initialize test data
        test_model = lambda x: [1 for _ in x]
        test_example = InputData(
            text="This is a test sentence.", model=test_model
        )
        test_n = 5
        test_neighbourhood = sampler.sample(test_example, test_model, test_n)
        test_candidate = CandidateAnchor()
        test_state = BeamState(
            test_neighbourhood,
            {},
            np.ones((test_n, 6)),
            6,
            test_example,
            test_model,
        )
        # run test
        candidate, state = sampler.sample_candidate_with_state(
            test_candidate, test_state
        )
        assert isinstance(
            candidate, CandidateAnchor
        ), f"Expected CandidateAnchor, got {type(candidate)}."
        assert isinstance(
            state, BeamState
        ), f"Expected BeamState, got {type(state)}."
