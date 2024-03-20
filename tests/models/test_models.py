import os

import omegaconf
from hydra.utils import instantiate

from globalanchors import constants
from globalanchors.models import BaseModel

MODEL_CONFIG_DIR = constants.HYDRA_CONFIG_PATH / "model"


def test_instantiate_models():
    """Test we can instantiate all models."""
    for t in os.listdir(MODEL_CONFIG_DIR):
        config_path = MODEL_CONFIG_DIR / t
        cfg = omegaconf.OmegaConf.load(config_path)

        model = instantiate(cfg)

        assert model, f"Model {t} not instantiated!"
        assert isinstance(model, BaseModel)


def test_train_models():
    """Test we can train all models."""
    for t in os.listdir(MODEL_CONFIG_DIR):
        config_path = MODEL_CONFIG_DIR / t
        cfg = omegaconf.OmegaConf.load(config_path)

        model = instantiate(cfg)
        # initialize training data
        data = ["This is a test sentence.", "This is another test sentence."]
        labels = [0, 1]
        dataset = {
            "train_data": data,
            "train_labels": labels,
            "val_data": data,
            "val_labels": labels,
            "test_data": data,
            "test_labels": labels,
        }
        model.train(dataset)
        try:
            model.model.predict(model.vectorizer.transform(data))
        except Exception as e:
            assert False, f"Model {t} failed to predict after training: {e}."
        assert True


def test_predict_models():
    """Test we can predict an expected result with all models."""
    for t in os.listdir(MODEL_CONFIG_DIR):
        config_path = MODEL_CONFIG_DIR / t
        cfg = omegaconf.OmegaConf.load(config_path)

        model = instantiate(cfg)
        # initialize training data
        data = ["This is a test sentence.", "This is another test sentence."]
        labels = [0, 1]
        dataset = {
            "train_data": data,
            "train_labels": labels,
            "val_data": data,
            "val_labels": labels,
            "test_data": data,
            "test_labels": labels,
        }
        model.train(dataset)

        test_input = ["This is a test sentence."]
        expected_result = [0]
        try:
            result = model(test_input)
            assert (
                result == expected_result
            ), f"Model {t} predicted {result} when {expected_result} was expected."
        except Exception as e:
            assert False, f"Model {t} errored when predicting: {e}."
