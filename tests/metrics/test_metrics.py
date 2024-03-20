import omegaconf
from hydra.utils import instantiate

from globalanchors import constants
from globalanchors.metrics import (
    calculate_local_metrics,
    calculate_global_metrics,
)
from globalanchors.types import LocalMetrics, GlobalMetrics

GLOBAL_CONFIG_FILE = (
    constants.HYDRA_CONFIG_PATH / "combined" / "submodular.yaml"
)
LOCAL_CONFIG_FILE = constants.HYDRA_CONFIG_PATH / "local" / "anchors.yaml"


def test_local_metrics():
    """Test local metrics run."""
    # initialize test data
    test_explainer = instantiate(omegaconf.OmegaConf.load(LOCAL_CONFIG_FILE))
    test_dataset = [
        "This is a test sentence.",
        "This is another test sentence.",
    ]
    test_model = lambda x: [1 for _ in x]
    # run test
    metrics = calculate_local_metrics(test_explainer, test_dataset, test_model)
    assert metrics, "Metrics not returned!"
    assert isinstance(
        metrics, LocalMetrics
    ), f"Expected LocalMetrics, got {type(metrics)}."


def test_global_metrics():
    """Test global metrics run."""
    # initialize test data
    local_explainer = instantiate(omegaconf.OmegaConf.load(LOCAL_CONFIG_FILE))
    test_explainer = instantiate(omegaconf.OmegaConf.load(GLOBAL_CONFIG_FILE))
    test_dataset = [
        "This is a test sentence.",
        "This is another test sentence.",
    ]
    test_model = lambda x: [1 for _ in x]
    test_explainer.train(local_explainer, test_dataset, test_model)
    # run test
    metrics = calculate_global_metrics(test_explainer, test_dataset)
    assert metrics, "Metrics not returned!"
    assert isinstance(
        metrics, GlobalMetrics
    ), f"Expected GlobalMetrics, got {type(metrics)}."
