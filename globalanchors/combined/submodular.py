from typing import List, override

from loguru import logger

from globalanchors.combined.base import GlobalAnchors
from globalanchors.local.anchors import TextAnchors
from globalanchors.types import ExplainerOutput, Model


class SubmodularPick(GlobalAnchors):
    """Implements submodular pick for selecting rule subset (i.e., greedy set cover)."""

    def __init__(
        self, explainer: TextAnchors, data: List[str], num_rules: int = 5
    ):
        self.num_rules = num_rules
        super().__init__(explainer, data)

    @override
    def combine_rules(self, model: Model) -> List[ExplainerOutput]:
        # generate explanations
        explanations = []
        for text in self.data:
            # check for cached output
            if text in self.explaination_cache:
                explanations.append(self.explaination_cache[text])
            else:
                expl = self.explainer.explain(text, model)
                self.explaination_cache[text] = expl
                explanations.append(expl)
        # calculate explanation coverage
        covered = {}
        for i, expl in explanations:
            covered[i] = set(
                [
                    j
                    for j, text in self.data
                    if len(expl.explanation) > 0
                    and all([feat in text for feat in expl.explanation])
                ]
            )
        # choose explanations that maximize coverage
        chosen = []
        current_covered = set()
        for i in range(self.num_rules):
            best = (-1, -1)  # (index, gain)
            for j in covered:
                gain = len(current_covered.union(covered[j]))
                if gain > best[1]:
                    best = (j, gain)
            all_covered = all_covered.union(covered[best[0]])
            logger.debug(
                f"Chose explanation {best[0]} with coverage {best[1] / len(self.data)}"
            )
            chosen.append(explanations[best[0]])
        return chosen
