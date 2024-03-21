"""Base implementation of a local-to-global rule-based aggregator."""

from abc import ABC, abstractmethod
from typing import List, Union

from loguru import logger
import numpy as np

from globalanchors.local.anchors import TextAnchors
from globalanchors.types import ExplainerOutput, GlobalExplainerOutput, Model


class GlobalAnchors(ABC):
    def __init__(self, num_rules: int = 5):
        self.num_rules = num_rules
        self.explanation_cache = {}
        self.data = None
        self.model = None
        self.rules = None

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
        self.rules = self.combine_rules()

    def explain(self, example: Union[str, bytes]) -> GlobalExplainerOutput:
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
        # check that explainer has been trained
        assert (
            self.rules is not None
        ), "Explainer must be trained with <explainer.train()> before explaining."
        # find valid rules for this example
        valid_explanations = [
            expl
            for expl in self.rules
            if all([feat in example for feat in expl["explanation"]])
        ]
        # predict using the model
        if len(valid_explanations) == 0:
            # randomly predict
            prediction = np.random.choice(2)
            used_rule = None
        else:
            # choose the valid rule with the highest precision (i.e., accuracy)
            used_rule = max(valid_explanations, key=lambda x: x["precision"])
            prediction = used_rule["prediction"]
        return {
            "example": example,
            "explanations": valid_explanations,
            "rule_used": used_rule,
            "prediction": prediction,
        }
