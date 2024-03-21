from typing import List

from loguru import logger
import numpy as np

from globalanchors.combined.base import GlobalAnchors
from globalanchors.anchor_types import ExplainerOutput


class RuleScoring(GlobalAnchors):
    """Implements rule-based scoring for selecting rule subset from https://kdd.isti.cnr.it/publications/global-explanations-local-scoring."""

    def __init__(self, num_rules: int = 5):
        super().__init__(num_rules)

    def _rule_relevance_scores(
        self, explanations: List[ExplainerOutput]
    ) -> List[float]:
        n_rules, n_data = len(explanations), len(self.data)
        # calculate coverage matrix (C[i, j] means rule i covers example j)
        coverage_matrix = np.zeros((n_rules, n_data))
        for i, expl in enumerate(explanations):
            covered = np.array(
                [
                    (
                        1
                        if all([feat in text for feat in expl["explanation"]])
                        else 0
                    )
                    for text in self.data
                ],
                dtype=bool,
            )
            coverage_matrix[i] = covered
        # maximize coverage
        coverage_score = coverage_matrix.sum(axis=1) / n_data
        # minimize repeats/conflicts
        pattern_coverage_score = 1 - np.sum(coverage_matrix, axis=0) / n_rules
        rule_coverage_score = np.matmul(
            coverage_matrix, pattern_coverage_score
        )
        rule_coverage_score = (
            rule_coverage_score / rule_coverage_score.max()
        )  # normalize
        rule_coverage_score = np.vectorize(lambda x: 0 if np.isnan(x) else x)(
            rule_coverage_score
        )  # handle NaNs
        # combine
        return (coverage_score + rule_coverage_score / 2).tolist()

    # override
    def combine_rules(self) -> List[ExplainerOutput]:
        # generate explanations
        logger.info("Generating explanations...")
        explanations = []
        for text in self.data:
            # check for cached output
            if text in self.explanation_cache:
                explanations.append(self.explanation_cache[text])
            else:
                expl = self.explainer.explain(text)
                self.explanation_cache[text] = expl
                explanations.append(expl)
        # calculate scores
        logger.info("Calculating rule relevance scores...")
        scores = self._rule_relevance_scores(explanations)
        # choose explanations with the highest score
        chosen_idx = np.argsort(scores)[-self.num_rules :]
        return [explanations[i] for i in chosen_idx]
