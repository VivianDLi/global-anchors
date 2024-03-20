import os

import omegaconf
from hydra.utils import instantiate

from globalanchors import constants
from globalanchors.combined.base import GlobalAnchors
from globalanchors.types import GlobalExplainerOutput

GLOBAL_CONFIG_DIR = constants.HYDRA_CONFIG_PATH / "combined"
LOCAL_CONFIG_FILE = constants.HYDRA_CONFIG_PATH / "local" / "anchors.yaml"


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
        config_path = GLOBAL_CONFIG_DIR / t
        cfg = omegaconf.OmegaConf.load(config_path)

        explainer = instantiate(cfg)
        # initialize test data
        test_explainer = instantiate(
            omegaconf.OmegaConf.load(LOCAL_CONFIG_FILE)
        )
        test_data = [
            "This is a test sentence.",
            "This is another test sentence.",
        ]
        test_model = lambda x: [1 for _ in x]
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
        config_path = GLOBAL_CONFIG_DIR / t
        cfg = omegaconf.OmegaConf.load(config_path)

        explainer = instantiate(cfg)
        # initialize test data
        test_explainer = instantiate(
            omegaconf.OmegaConf.load(LOCAL_CONFIG_FILE)
        )
        test_data = [
            "This is a test sentence.",
            "This is another test sentence.",
        ]
        test_model = lambda x: [1 for _ in x]
        test_example = "This is a test sentence."
        explainer.train(test_explainer, test_data, test_model)
        # run tests
        explanations = explainer.explain(test_example)
        assert explanations, "Explanations not returned!"
        assert isinstance(
            explanations, GlobalExplainerOutput
        ), f"Expected GlobalExplainerOutput, got {type(explanations)}."
