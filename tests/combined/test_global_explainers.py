import os

from loguru import logger
import omegaconf
from hydra.utils import instantiate

from globalanchors import constants
from globalanchors.combined.base import GlobalAnchors

GLOBAL_CONFIG_DIR = constants.HYDRA_CONFIG_PATH / "combined"
LOCAL_CONFIG_FILE = constants.HYDRA_CONFIG_PATH / "local" / "anchors.yaml"
SAMPLER_CONFIG_FILE = constants.HYDRA_CONFIG_PATH / "sampler" / "unk.yaml"


def test_instantiate_global():
    """Test we can instantiate local explainer."""
    for t in os.listdir(GLOBAL_CONFIG_DIR):
        config_path = GLOBAL_CONFIG_DIR / t
        cfg = omegaconf.OmegaConf.load(config_path)

        explainer = instantiate(cfg)

        assert explainer, f"Global Explainer not instantiated!"
        assert isinstance(explainer, GlobalAnchors)


def test_train():
    """Test train function runs."""
    for t in os.listdir(GLOBAL_CONFIG_DIR):
        logger.info("Testing combiner: {t}")
        config_path = GLOBAL_CONFIG_DIR / t
        cfg = omegaconf.OmegaConf.load(config_path)

        explainer = instantiate(cfg)
        # initialize test data
        test_explainer = instantiate(
            omegaconf.OmegaConf.load(LOCAL_CONFIG_FILE)
        )
        test_sampler = instantiate(
            omegaconf.OmegaConf.load(SAMPLER_CONFIG_FILE)
        )
        test_model = lambda x: [1 for _ in x]
        test_explainer.set_functions(test_sampler, test_model)
        test_data = [
            "This is a test sentence.",
            "This is another test sentence.",
        ]
        # run tests
        explainer.train(test_explainer, test_data, test_model)
        assert explainer.data == test_data, "Data not set!"
        assert explainer.model == test_model, "Model not set!"
        assert hasattr(explainer, "rules") and isinstance(
            explainer.rules, list
        ), "Rules not set!"


def test_explain():
    """Test explain function runs."""
    for t in os.listdir(GLOBAL_CONFIG_DIR):
        logger.info("Testing combiner: {t}")
        config_path = GLOBAL_CONFIG_DIR / t
        cfg = omegaconf.OmegaConf.load(config_path)

        explainer = instantiate(cfg)
        # initialize test data
        test_explainer = instantiate(
            omegaconf.OmegaConf.load(LOCAL_CONFIG_FILE)
        )
        test_sampler = instantiate(
            omegaconf.OmegaConf.load(SAMPLER_CONFIG_FILE)
        )
        test_model = lambda x: [1 for _ in x]
        test_explainer.set_functions(test_sampler, test_model)
        test_data = [
            "This is a test sentence.",
            "This is another test sentence.",
        ]
        test_example = "This is a test sentence."
        explainer.train(test_explainer, test_data, test_model)
        expected_keys = set(
            ["example", "explanations", "rule_used", "prediction"]
        )
        # run tests
        explanation = explainer.explain(test_example)
        assert explanation, "Explanations not returned!"

        current_keys = set(explanation.keys())
        assert (
            current_keys == expected_keys
        ), f"Output for explainer {cfg} is missing keys {expected_keys - current_keys}."
