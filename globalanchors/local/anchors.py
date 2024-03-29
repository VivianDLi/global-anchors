"""Base anchor functions"""

from typing import Dict, List, Tuple, Union
import numpy as np

from loguru import logger

from globalanchors.local.neighbourhood.base import NeighbourhoodSampler
from globalanchors.local.utils import bernoulli_lb, bernoulli_ub, compute_beta
from globalanchors.utils import normalize_feature_indices
from globalanchors.anchor_types import (
    InputData,
    CandidateAnchor,
    BeamState,
    ExplainerOutput,
    Model,
)


class TextAnchors:
    def __init__(
        self,
        confidence_threshold: float = 0.95,
        epsilon: float = 0.1,
        delta: float = 0.05,
        batch_size: int = 10,
        beam_size: int = 4,
        min_start_samples: int = 0,
        max_anchor_size: int = None,
        stop_on_first: bool = False,
        coverage_samples: int = 10000,
        log_interval: int = 500,
    ):
        """Generates local explanations for examples using Anchors.

        Args:
            confidence_threshold (float, optional): Precision threshold for candidate anchors. Defaults to 0.95.
            epsilon (float, optional): Confidence threshold. Defaults to 0.1.
            delta (float, optional): Delta hyperparameter for estimating confidence. Defaults to 0.05.
            batch_size (int, optional): Number of samples to generate each time. Defaults to 10.
            beam_size (int, optional): Number of candidates to keep track of. Defaults to 4.
            min_start_samples (int, optional): Initial sample size. Defaults to 15.
            max_anchor_size (int, optional): Maximum number of features in an anchor. None equates to the input length. Defaults to None.
            stop_on_first (bool, optional): Stops on the first candidate anchor with high enough precision. Defaults to False.
            coverage_samples (int, optional): Number of samples to use to calcualte coverage. Defaults to 10000.
            log_interval (int, optional): When to log confidence estimations. Defaults to 5.
        """
        self.sampler = None
        self.model = None
        self.confidence_threshold = confidence_threshold
        self.epsilon = epsilon
        self.delta = delta
        self.batch_size = batch_size
        self.beam_size = beam_size
        self.min_start_samples = min_start_samples
        self.max_anchor_size = max_anchor_size
        self.stop_on_first = stop_on_first
        self.coverage_samples = coverage_samples
        self.log_interval = log_interval

        self.explanation_cache = {}

    def _get_candidate_anchors(
        self, past_candidates: List[CandidateAnchor], state: BeamState
    ) -> List[CandidateAnchor]:
        """Generates <length + 1> anchors for the current candidate set.

        Args:
            past_candidates (List[CandidateAnchor]): List of <length> anchors from a past iteration.
            state (BeamState): Current state of the beam-search, including neighbourhood data.

        Returns:
            List[CandidateAnchor]: List of new anchors.
        """
        feature_indices = range(state.n_features)
        # initialize an empty anchor for the base case of no previous candidates
        if len(past_candidates) == 0:
            past_candidates = [
                CandidateAnchor(
                    coverage=1,
                    prediction=state.example.label,
                    num_samples=0,
                    num_positives=0,
                    coverage_idx=set(range(state.coverage_data.shape[0])),
                    data_idx=set(),
                )
            ]
        # calculate new anchors by adding each feature to existing candidates and update state
        new_candidates: List[CandidateAnchor] = []
        for f_i in feature_indices:
            feat_cand = state.get_anchor([f_i])
            for p_cand in past_candidates:
                new_feat_indices = p_cand.feat_indices.union(set([f_i]))
                # check for duplicates
                if len(new_feat_indices) == len(
                    p_cand.feat_indices
                ) or normalize_feature_indices(new_feat_indices) in [
                    normalize_feature_indices(x.feat_indices)
                    for x in new_candidates
                ]:
                    continue
                new_feats = set(
                    [state.example.tokens[i] for i in new_feat_indices]
                )
                # combine coverage indices and recalculate coverage
                new_coverage_indices = p_cand.coverage_idx.intersection(
                    feat_cand.coverage_idx
                )
                new_coverage = (
                    float(len(new_coverage_indices))
                    / state.coverage_data.shape[0]
                )
                # combine data indices and recalculate precision
                prev_indices = np.array(list(p_cand.data_idx))
                # account for the base case of no previous indices
                if len(prev_indices) == 0:
                    new_candidates.append(
                        CandidateAnchor(
                            new_feats,
                            new_feat_indices,
                            0,
                            new_coverage,
                            state.example.label,
                            0,
                            0,
                            set(),
                            new_coverage_indices,
                        )
                    )
                    continue
                prev_data = state.neighbourhood.data[prev_indices]
                feat_present = np.where(prev_data[:, f_i] == 1)[0]
                new_covered_indices = set(prev_indices[feat_present])
                new_num_samples = float(len(new_covered_indices))
                new_num_positives = float(
                    state.neighbourhood.labels[list(new_covered_indices)].sum()
                )
                # handle case of no samples
                new_precision = (
                    new_num_positives / new_num_samples
                    if new_num_samples > 0
                    else 0
                )
                # new metadata
                new_prediction = state.example.label
                new_candidates.append(
                    CandidateAnchor(
                        new_feats,
                        new_feat_indices,
                        new_precision,
                        new_coverage,
                        new_prediction,
                        new_num_samples,
                        new_num_positives,
                        new_covered_indices,
                        new_coverage_indices,
                    )
                )
        return new_candidates

    def _get_best_candidates(
        self, candidates: List[CandidateAnchor], state: BeamState
    ) -> Tuple[List[CandidateAnchor], List[int]]:
        """Finds the best candidate anchors using the KL-LUCB algorithm (https://proceedings.mlr.press/v30/Kaufmann13.html).

        Args:
            state (BeamState): Current state of the beam-search, including current anchors and neighbourhood data.

        Returns:
            List[int]: List of the indices of the best candidate anchors sorted by precision.
        """
        # initialize bounds arrays
        n_features = len(candidates)
        # handle length 1 case
        if n_features == 1:
            return candidates, [0]
        n_samples = np.array([cand.num_samples for cand in candidates])
        n_positives = np.array([cand.num_positives for cand in candidates])
        upper_bounds = np.zeros(n_features)
        lower_bounds = np.zeros(n_features)
        top_n = min(n_features - 1, self.beam_size)
        # check candidates without samples, and generate samples
        for i in np.where(n_samples == 0)[0]:
            candidates[i], state = self.sampler.sample_candidate_with_state(
                candidates[i], state
            )
            n_samples[i] = candidates[i].num_samples
            n_positives[i] = candidates[i].num_positives
            logger.debug(
                f"Num Samples after correcting for candidate {i}: {n_samples}."
            )

        # define function to calculate bounds from array
        def _update_bounds(t, means):
            sorted_means = np.argsort(means)
            beta = compute_beta(n_features, t, self.delta)
            # splitting candidates into two sets to calculate lower and upper bounds
            J = sorted_means[-top_n:]
            not_J = sorted_means[:-top_n]
            for i in J:
                lower_bounds[i] = bernoulli_lb(means[i], beta / n_samples[i])
            for i in not_J:
                upper_bounds[i] = bernoulli_ub(means[i], beta / n_samples[i])
            lower_index = J[np.argmin(lower_bounds[J])]
            upper_index = not_J[np.argmax(upper_bounds[not_J])]
            return lower_index, upper_index

        ## bandit selection loop
        # initialize loop variables
        t = 1
        means = n_positives / n_samples  # precisions
        lower_i, upper_i = _update_bounds(t, means)
        diff = upper_bounds[upper_i] - lower_bounds[lower_i]
        while diff > self.epsilon:
            # logging (disabled for tests)
            if (t % self.log_interval) == 0:
                logger.info(
                    f"\nBest Stats: (i: {lower_i}, mean: {means[lower_i]:.1f}, n: {n_samples[lower_i]}, lb: {lower_bounds[lower_i]:.4f})\nWorst Stats: (i: {upper_i}, mean: {means[upper_i]:.1f}, n: {n_samples[upper_i]}, ub: {upper_bounds[upper_i]:.4f})\nDiff: {diff:.2f}"
                )
            # generate samples
            candidates[upper_i], state = (
                self.sampler.sample_candidate_with_state(
                    candidates[upper_i], state, n=self.batch_size
                )
            )
            candidates[lower_i], state = (
                self.sampler.sample_candidate_with_state(
                    candidates[lower_i], state, n=self.batch_size
                )
            )
            n_samples[upper_i] = candidates[upper_i].num_samples
            n_positives[upper_i] = candidates[upper_i].num_positives
            n_samples[lower_i] = candidates[lower_i].num_samples
            n_positives[lower_i] = candidates[lower_i].num_positives
            means = n_positives / n_samples
            # update loop variables
            t += 1
            lower_i, upper_i = _update_bounds(t, means)
            diff = upper_bounds[upper_i] - lower_bounds[lower_i]
        sorted_means = np.argsort(means)
        return candidates, sorted_means[-top_n:]

    def _beam_search(self, example: str) -> CandidateAnchor:
        """Performs beam-search to greedily find the best anchors as explanations.

        Args:
            example (str): String of text to generate an explanation for.

        Returns:
            CandidateAnchor: Best explanation found for the current model and example.
        """
        example_data = InputData(example, self.model)
        logger.info("Generating coverage data...")
        # generate massive amount of data to estimate coverage
        coverage_data = self.sampler.sample(
            example_data,
            self.model,
            n=self.coverage_samples,
            compute_labels=False,
            compute_coverage=True,
        ).data
        logger.info("Initializing beam-search...")
        # generate neighbourhood samples
        neighbourhood = self.sampler.sample(
            example_data,
            self.model,
            n=max(1, self.min_start_samples),
        )
        # evaluate lower bound precision of candidates
        mean = neighbourhood.labels.mean()
        beta = np.log(1.0 / self.delta)
        lower_bound = bernoulli_lb(mean, beta / neighbourhood.data.shape[0])
        # generate neighbourhood samples until null anchor is valid (i.e., model accuracy is high enough)
        while (
            mean > self.confidence_threshold
            and lower_bound < self.confidence_threshold - self.epsilon
        ):
            neighbourhood = self.sampler.sample(
                example_data,
                self.model,
                n=self.batch_size,
                neighbourhood=neighbourhood,
            )
            mean = neighbourhood.labels.mean()
            lower_bound = bernoulli_lb(
                mean, beta / neighbourhood.data.shape[0]
            )
        # if the null anchor meets the threshold, quick stop
        if lower_bound > self.confidence_threshold:
            logger.info("Stopping beam-search early with null anchor rule.")
            pred_label = neighbourhood.labels[
                np.argmax(
                    np.unique(neighbourhood.labels, return_counts=True)[1]
                )
            ]
            return CandidateAnchor(
                precision=mean,
                coverage=1,
                prediction=pred_label,
                num_samples=neighbourhood.data.shape[0],
            )
        ## enter main beam search loop
        logger.info("Starting beam-search...")
        # pre-allocate neighbourhood data memory
        neighbourhood.prealloc_size = self.batch_size * 10000
        neighbourhood.reallocate()
        # initial loop state
        state = BeamState(
            neighbourhood=neighbourhood,
            anchors={},
            coverage_data=coverage_data,
            n_features=neighbourhood.data.shape[1],
            example=example_data,
            model=self.model,
        )
        state.initialize_features()
        current_size = 1
        best_anchors_per_size: Dict[int, List[CandidateAnchor]] = {0: []}
        best_coverage = 0  # check for 0 coverage
        best_anchor = None
        if self.max_anchor_size is None:
            self.max_anchor_size = neighbourhood.data.shape[1]
        # start loop
        while current_size <= self.max_anchor_size:
            # generate new candidate anchors based on previous best anchors
            candidates = self._get_candidate_anchors(
                best_anchors_per_size[current_size - 1], state
            )
            # filter out candidates better than previous loop
            candidates = [
                anch for anch in candidates if anch.coverage > best_coverage
            ]
            if len(candidates) == 0:
                logger.info(
                    f"Stopping beam-search early at anchor size {current_size} / {self.max_anchor_size} from no more candidates."
                )
                break
            # select the new best candidates with confidence bounds
            candidates, candidate_indices = self._get_best_candidates(
                candidates, state
            )
            best_anchors_per_size[current_size] = [
                candidates[i] for i in candidate_indices
            ]
            # add candidates to state dict
            for candidate in best_anchors_per_size[current_size]:
                state.set_anchor(candidate.feat_indices, candidate)
            logger.info(
                f"Best of anchor size {current_size}: {[cand.feats for cand in best_anchors_per_size[current_size]]}"
            )
            # choose beam_size new best candidates based on coverage
            early_stop = False
            logger.debug(f"Candidates: {best_anchors_per_size[current_size]}.")
            for i, candidate in zip(
                candidate_indices, best_anchors_per_size[current_size]
            ):
                beta = np.log(
                    1.0
                    / (
                        self.delta
                        / (1 + (self.beam_size - 1) * state.n_features)
                    )
                )
                mean = candidate.num_positives / candidate.num_samples
                lower_bound = bernoulli_lb(mean, beta / candidate.num_samples)
                upper_bound = bernoulli_ub(mean, beta / candidate.num_samples)
                coverage = candidate.coverage
                logger.debug(
                    f"Mean, LB, UB, and Coverage for the {i}th anchor {candidate.feats}: {mean:0.2f}, {lower_bound:0.2f}, {upper_bound:0.2f}, {coverage:0.2f}."
                )
                # until confidence threshold is met, continue sampling neighbourhood and updating LB and UB
                while (
                    mean >= self.confidence_threshold
                    and lower_bound < self.confidence_threshold - self.epsilon
                ) or (
                    mean < self.confidence_threshold
                    and upper_bound >= self.confidence_threshold + self.epsilon
                ):
                    candidate, state = (
                        self.sampler.sample_candidate_with_state(
                            candidate, state, self.batch_size
                        )
                    )
                    mean = candidate.num_positives / candidate.num_samples
                    lower_bound = bernoulli_lb(
                        mean, beta / candidate.num_samples
                    )
                    upper_bound = bernoulli_ub(
                        mean, beta / candidate.num_samples
                    )
                logger.debug(
                    f"Mean, LB, and UB after updating for the {i}th anchor {candidate.feats}: {mean:0.2f}, {lower_bound:0.2f}, {upper_bound:0.2f}."
                )
                if (
                    mean >= self.confidence_threshold
                    and lower_bound > self.confidence_threshold - self.epsilon
                ):
                    logger.info(
                        f"Found eligible anchor {candidate.feats} with coverage {coverage:0.2f} and current max coverage is {best_coverage:0.2f}."
                    )
                    if coverage > best_coverage:
                        best_coverage = coverage
                        best_anchor = candidate
                        # exit loop early if maximum coverage found or explainer set to stop on first candidate
                        if best_coverage == 1 or self.stop_on_first:
                            logger.info(
                                f"Stopping beam-search early with best coverage {best_coverage}."
                            )
                            early_stop = True
            if early_stop:
                break
            current_size += 1
        # Check for no best anchor found
        if best_anchor is None:
            logger.info(
                "Could not find an anchor satisfying requirements. Now searching for the best anchor found so far."
            )
            all_candidates = [
                cand
                for cand_list in [
                    best_anchors_per_size[size]
                    for size in best_anchors_per_size
                ]
                for cand in cand_list
            ]
            _, candidate_indices = self._get_best_candidates(
                all_candidates, state
            )
            best_anchor = all_candidates[candidate_indices[0]]
        return best_anchor

    def set_functions(self, sampler: NeighbourhoodSampler, model: Model):
        self.sampler = sampler
        self.model = model
        self.explanation_cache = {}

    def explain(self, example: Union[str, bytes]) -> ExplainerOutput:
        """Creates local explanations for a textual example using Anchors for a given model.

        Args:
            example (str): Textual example to generate explanations for.
            model (Model): ML Model to explain.
        """
        assert (
            self.sampler is not None and self.model is not None
        ), "Sampler and Model must be set with <set_functions> before explaining."
        # optionally decode a byte input
        if type(example) == bytes:
            logger.debug("Decoding byte string example.")
            example = example.decode()
        assert (
            type(example) == str
        ), f"Explainer input must be either a string or a byte array. Input was instead a {type(example)}."
        # check for cached explanation
        if example in self.explanation_cache:
            return self.explanation_cache[example]
        else:
            exp = self._beam_search(example)
            new_result = {
                "example": example,
                "explanation": exp.feats,
                "precision": exp.precision,
                "coverage": exp.coverage,
                "prediction": exp.prediction,
                "num_samples": exp.num_samples,
            }
            self.explanation_cache[example] = new_result
            return new_result
