import omegaconf
from hydra.utils import instantiate

from globalanchors import constants
from globalanchors.metrics import (
    calculate_local_metrics,
    calculate_global_metrics,
)

GLOBAL_CONFIG_FILE = (
    constants.HYDRA_CONFIG_PATH / "combined" / "submodular.yaml"
)
LOCAL_CONFIG_FILE = constants.HYDRA_CONFIG_PATH / "local" / "anchors.yaml"
SAMPLER_CONFIG_FILE = constants.HYDRA_CONFIG_PATH / "sampler" / "unk.yaml"


def test_local_metrics():
    """Test local metrics run."""
    # initialize test data
    test_explainer = instantiate(omegaconf.OmegaConf.load(LOCAL_CONFIG_FILE))
    test_sampler = instantiate(omegaconf.OmegaConf.load(SAMPLER_CONFIG_FILE))
    test_model = lambda x: [1 for _ in x]
    test_explainer.set_functions(test_sampler, test_model)
    test_dataset = [
        "This is a test sentence.",
        "This is another test sentence.",
    ]
    expected_keys = set(
        [
            "rule_length",
            "coverage",
            "precision",
            "f1",
            "num_samples",
            "time_taken",
        ]
    )
    # run test
    metrics = calculate_local_metrics(
        test_explainer, test_dataset, log_to_wandb=False
    )
    assert metrics, "Metrics not returned!"

    current_keys = set(metrics.keys())
    assert (
        current_keys == expected_keys
    ), f"Output for local metrics is missing keys {expected_keys - current_keys}."


def test_global_metrics():
    """Test global metrics run."""
    # initialize test data
    local_explainer = instantiate(omegaconf.OmegaConf.load(LOCAL_CONFIG_FILE))
    test_sampler = instantiate(omegaconf.OmegaConf.load(SAMPLER_CONFIG_FILE))
    test_model = lambda x: [1 for _ in x]
    local_explainer.set_functions(test_sampler, test_model)
    test_explainer = instantiate(omegaconf.OmegaConf.load(GLOBAL_CONFIG_FILE))
    test_dataset = [
        "This is a test sentence.",
        "This is another test sentence.",
    ]
    test_explainer.train(local_explainer, test_dataset)
    expected_keys = set(
        [
            "global_rule_length",
            "global_rule_coverage",
            "global_rule_precision",
            "global_rule_f1",
            "average_valid_rules",
            "rule_length",
            "accuracy",
            "coverage",
        ]
    )
    # run test
    metrics = calculate_global_metrics(
        test_explainer, test_dataset, log_to_wandb=False
    )
    assert metrics, "Metrics not returned!"

    current_keys = set(metrics.keys())
    assert (
        current_keys == expected_keys
    ), f"Output for global metrics is missing keys {expected_keys - current_keys}."
