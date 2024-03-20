"""Base implementation of a local-to-global rule-based aggregator."""

from abc import ABC, abstractmethod
from typing import List, Union

from loguru import logger

from globalanchors.local.anchors import TextAnchors
from globalanchors.types import ExplainerOutput, Model


class GlobalAnchors(ABC):
    def __init__(self, num_rules: int = 5):
        self.num_rules = num_rules
        self.explanation_cache = {}

    @abstractmethod
    def combine_rules(self, model: Model) -> List[ExplainerOutput]:
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
            if all([feat in example for feat in expl.feats])
        ]
