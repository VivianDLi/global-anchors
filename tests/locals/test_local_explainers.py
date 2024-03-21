import omegaconf
from hydra.utils import instantiate

from globalanchors import constants
from globalanchors.local.anchors import TextAnchors

LOCAL_CONFIG_FILE = constants.HYDRA_CONFIG_PATH / "local" / "anchors.yaml"
SAMPLER_CONFIG_FILE = constants.HYDRA_CONFIG_PATH / "sampler" / "unk.yaml"


def test_instantiate_local():
    """Test we can instantiate local explainer."""
    cfg = omegaconf.OmegaConf.load(LOCAL_CONFIG_FILE)

    explainer = instantiate(cfg)

    assert explainer, f"Local Explainer not instantiated!"
    assert isinstance(explainer, TextAnchors)


def test_explain_null_anchor():
    """Test explain function runs."""
    cfg = omegaconf.OmegaConf.load(LOCAL_CONFIG_FILE)
    explainer = instantiate(cfg)
    sampler = instantiate(omegaconf.OmegaConf.load(SAMPLER_CONFIG_FILE))
    test_model = lambda x: [1 for _ in x]
    explainer.set_functions(sampler, test_model)
    # initialize test data
    test_example = "This is a test sentence."
    expected_keys = set(
        [
            "example",
            "explanation",
            "precision",
            "coverage",
            "prediction",
            "num_samples",
        ]
    )
    # run test
    explanation = explainer.explain(test_example, test_model)
    assert explanation, "Explanation not returned!"

    current_keys = set(explanation.keys())
    assert (
        current_keys == expected_keys
    ), f"Output for explainer {cfg} is missing keys {expected_keys - current_keys}."


def test_explain_complex_anchor():
    """Test explain function runs."""
    cfg = omegaconf.OmegaConf.load(LOCAL_CONFIG_FILE)
    explainer = instantiate(cfg)
    sampler = instantiate(omegaconf.OmegaConf.load(SAMPLER_CONFIG_FILE))
    test_model = lambda xs: [1 if "test" in x else 0 for x in xs]
    explainer.set_functions(sampler, test_model)
    # initialize test data
    test_example = "This is a test sentence."
    expected_feature = "test"
    expected_keys = set(
        [
            "example",
            "explanation",
            "precision",
            "coverage",
            "prediction",
            "num_samples",
        ]
    )
    # run test
    explanation = explainer.explain(test_example, test_model)
    assert explanation, "Explanation not returned!"

    current_keys = set(explanation.keys())
    assert (
        current_keys == expected_keys
    ), f"Output for explainer {cfg} is missing keys {expected_keys - current_keys}."
    assert (
        expected_feature in explanation["explanation"]
    ), f"Expected feature {expected_feature} as anchor, but got {explanation['explanation']} instead."
