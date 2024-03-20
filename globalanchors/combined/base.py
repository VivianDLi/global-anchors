"""Base implementation of a local-to-global rule-based aggregator."""

from abc import ABC, abstractmethod
from typing import List, Union

from loguru import logger
import numpy as np

from globalanchors.local.anchors import TextAnchors
from globalanchors.types import ExplainerOutput, Model


class GlobalAnchors(ABC):
    def __init__(self, num_rules: int = 5):
        self.num_rules = num_rules
        self.explanation_cache = {}

    @abstractmethod
    def combine_rules(self) -> List[ExplainerOutput]:
        """Generates a subset of global rules for a model from a set of local rules created by running an explainer on a dataset.

        Args:
            model (Model): _description_

        Returns:
            List[ExplainerOutput]: _description_
        """
        return NotImplementedError

    def train(
        self, explainer: TextAnchors, data: List[str], model: Model
    ) -> None:
        """Train the global explainer on a dataset and local explainer.

        Args:
            explainer (TextAnchors): Local explainer to use.
            data (List[str]): Training data to use.
        """
        self.explanation_cache = {}
        self.explainer = explainer
        self.data = data
        self.model = model

    def explain(self, example: Union[str, bytes]) -> List[ExplainerOutput]:
        """Return all global explanations satisfying a textual example for a given model.

        Args:
            example (str): Textual example to generate explanations for.
            model (Model): ML Model to explain.
        """
        # optionally decode a byte input
        if type(example) == bytes:
            logger.debug("Decoding byte string example.")
            example = example.decode()
        assert (
            type(example) == str
        ), f"Explainer input must be either a string or a byte array. Input was instead a {type(example)}."
        # generate global rules for this model
        rules = self.combine_rules()
        # find valid rules for this example
        return [
            expl
            for expl in rules
            if all([feat in example for feat in expl["feats"]])
        ]

    def predict(self, example: Union[str, bytes]) -> int:
        """Predict the class of an example using the model.

        Args:
            example (Union[str, bytes]): Example to predict the class of.

        Returns:
            int: Predicted class of the example.
        """
        if type(example) == bytes:
            logger.debug("Decoding byte string example.")
            example = example.decode()
        assert (
            type(example) == str
        ), f"Predict input must be either a string or a byte array. Input was instead a {type(example)}."
        # find valid global rules
        valid_explanations = self.explain(example)
        # predict using the model
        if len(valid_explanations) == 0:
            # randomly predict
            return np.random.choice(2)
        # choose the valid rule with the highest precision (i.e., accuracy)
        return max(valid_explanations, key=lambda x: x["precision"])[
            "prediction"
        ]
