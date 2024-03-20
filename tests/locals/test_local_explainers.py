import numpy as np
import omegaconf
from hydra.utils import instantiate

from globalanchors import constants
from globalanchors.local.anchors import TextAnchors
from globalanchors.types import (
    ExplainerOutput,
    InputData,
)

LOCAL_CONFIG_FILE = constants.HYDRA_CONFIG_PATH / "local" / "anchors.yaml"


def test_instantiate_local():
    """Test we can instantiate local explainer."""
    cfg = omegaconf.OmegaConf.load(LOCAL_CONFIG_FILE)

    explainer = instantiate(cfg)

    assert explainer, f"Local Explainer not instantiated!"
    assert isinstance(explainer, TextAnchors)


def test_explain():
    """Test explain function runs."""
    cfg = omegaconf.OmegaConf.load(LOCAL_CONFIG_FILE)

    explainer = instantiate(cfg)
    # initialize test data
    test_example = InputData(
        text="This is a test sentence.",
        tokens=np.array(["This", "is", "a", "test", "sentence", "."]),
        positions=np.array([0, 1, 2, 3, 4, 5]),
        label=0,
    )
    test_model = lambda x: [1 for _ in x]
    # run test
    explanation = explainer.explain(test_example, test_model)
    assert explanation, "Explanation not returned!"
    assert isinstance(
        explanation, ExplainerOutput
    ), f"Expected NeighbourhoodData, got {type(explanation)}."
