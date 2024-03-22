"""Base general implementation of neighbourhood generation for local explainers."""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
import numpy as np
import torch

from loguru import logger

from globalanchors.utils import exp_normalize
from globalanchors.anchor_types import (
    InputData,
    CandidateAnchor,
    Model,
    NeighbourhoodData,
    BeamState,
)


class NeighbourhoodSampler(ABC):
    use_generator: bool = True
    use_generator_probabilities: bool = False
    use_constant_probabilities: bool = False

    def __init__(
        self,
        use_generator: bool = True,
        use_generator_probabilities: bool = False,
        use_constant_probabilities: bool = False,
    ):
        self.use_generator = use_generator
        self.use_generator_probabilities = use_generator_probabilities
        self.use_constant_probabilities = use_constant_probabilities
        # for any models created afterward
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        if self.use_constant_probabilities:
            self.use_generator_probabilities = False
            logger.info(
                "For constant probabilities, generator probabilities cannot be used. Setting <use_generator_probabilities> to False."
            )
        if self.use_generator_probabilities:
            self.use_generator = True
            logger.info(
                "For generator probabilities to be used, generator must be enabled. Setting <use_generator> to True."
            )
        if self.use_generator:
            from transformers import pipeline

            # import BERT for masked text generation
            self.fill_masker = pipeline(
                "fill-mask",
                model="bert-base-cased",
                framework="pt",
                device=self.device,
            )

    def _get_unmasked_words(
        self, masked_texts: List[str], top_k: int = 25
    ) -> Dict[str, Tuple[List[str], List[float]]]:
        """Gets a list of generated words and probabilities for a batch of masked inputs (only supports one mask token)."""
        assert (
            self.use_generator
        ), "Text generator must be enabled to generate unmasked words."
        results = {}
        mask_results = self.fill_masker(masked_texts, top_k=top_k)
        for i, masked_input in enumerate(masked_texts):
            # mask_results[i] is a list of dictionaries (or mask_results is a list of dictionaries if no batch is used)
            words = (
                [result["token_str"] for result in mask_results[i]]
                if len(masked_texts) > 1
                else [result["token_str"] for result in mask_results]
            )
            probs = np.array(
                [result["score"] for result in mask_results[i]]
                if len(masked_texts) > 1
                else [result["score"] for result in mask_results]
            )
            probs = exp_normalize(probs).tolist()
            results[masked_input] = (words, probs)
        return results

    @abstractmethod
    def perturb_samples(
        self,
        example: InputData,
        data: np.ndarray,
        model: Model,
        compute_labels: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.array]:
        """Generates new text strings and labels in the neighbourhood of an example given the words in the example to replace.

        Args:
            example (InputData): Original text example of length w
            data (np.ndarray): Binary array of shape (n x w) with a 0 in the place of a word to replace.
            model (Model): Model use to generate classification labels for generated text.
            compute_labels (bool): If false, then returned prediction labels are a ones array.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.array]: Tuple of new text string array, fixed_indices, and new array of model prediction labels.
        """
        return NotImplementedError

    def sample(
        self,
        example: InputData,
        model: Model,
        n: int = 1,
        neighbourhood: NeighbourhoodData = None,
        compute_labels: bool = True,
        compute_coverage: bool = False,
    ) -> NeighbourhoodData:
        """Samples data from the neighbourhood of an example, optionally updating a given neighbourhood.

        Args:
            example (InputData): Example to generate samples around.
            model (_type_): Model to explain and use to predict labels.
            n (int, optional): Number of new examples to generate. Defaults to 1.
            neighbourhood (NeighbourhoodData, optional): Optional neighbourhood to add newly generated examples to. Defaults to None.
            compute_labels (bool, optional): Whether to also compute model labels for samples or not. Defaults to True.
            compute_coverage (bool, optional): Whether to skip word generation for coverage computation. Defaults to False.

        Returns:
            NeighbourhoodData: NeighbourhoodData object representing generated samples in the example neighbourhood.
        """
        # skip rest if only computing coverage
        if compute_coverage:
            # always generate coverage data with random probability
            probabilities = np.ones((n, len(example.tokens))) * 0.5
            random_sample = np.random.uniform(0, 1, size=probabilities.shape)
            new_data = (random_sample < probabilities).astype(int)
            return NeighbourhoodData(None, new_data, None)

        ## get new neighbourhood samples
        # calculate replacement probabilities for each feature
        n_features = len(example.tokens)
        probabilities = np.zeros(
            (n, n_features)
        )  # probability to keep feature
        if self.use_generator_probabilities:
            # usually swaps low-probability words (i.e., expect to move away from distribution boundaries)
            for i in range(n_features):
                masked_tokens = example.tokens.copy()
                masked_tokens[i] = self.fill_masker.tokenizer.mask_token
                masked_string = " ".join(masked_tokens)
                words, probs = self._get_unmasked_words(
                    [masked_string], top_k=500
                )[
                    masked_string
                ]  # get first (only) masked element
                probabilities[:, i] = min(
                    0.5, dict(zip(words, probs)).get(example.tokens[i], 0.01)
                )
        elif self.use_constant_probabilities:
            # when generating from an anchor, only keep anchor features fixed (e.g., when doing GA sampling)
            probabilities = np.zeros_like(probabilities)
        else:
            # randomly select each feature (i.e., probability = 0.5)
            probabilities = np.ones_like(probabilities) * 0.5

        # randomly choose either 0 and 1 for each sample/feature depending on probabilities
        random_sample = np.random.uniform(0, 1, size=probabilities.shape)
        new_data = (random_sample < probabilities).astype(int)

        new_string_data, new_data, new_labels = self.perturb_samples(
            example, new_data, model, compute_labels=compute_labels
        )
        ## updating existing (optional) neighbourhood or creating a new one
        if neighbourhood is not None:
            neighbourhood.update(new_string_data, new_data, new_labels)
            return neighbourhood
        else:
            return NeighbourhoodData(new_string_data, new_data, new_labels)

    def sample_candidate_with_state(
        self, candidate: CandidateAnchor, state: BeamState, n: int = 1
    ) -> Tuple[CandidateAnchor, BeamState]:
        """Samples data from the neighbourhood of an example, keeping features used by an anchor fixed.

        Updates state neighbourhood and adds/updates candidate anchor to state.

        Args:
            candidate (CandidateAnchor): Anchor used to fix features in place.
            state (BeamState): Current state of a beam-search, containing current neighbourhood, classification model, and example to generate around.
            n (int, optional): Number of new examples to generate. Defaults to 1.

        Returns:
            Tuple[CandidateAnchor, BeamState]: Tuple of updated candidate anchor and state
        """
        # calculate replacement probabilities for each feature not in the candidate anchor
        n_features = len(state.example.tokens)
        probabilities = np.zeros(
            (n, n_features)
        )  # probability to keep feature
        if self.use_generator_probabilities:
            # usually swaps low-probability words (i.e., expect to move away from distribution boundaries)
            for i in range(n_features):
                masked_tokens = state.example.tokens.copy()
                masked_tokens[i] = self.fill_masker.tokenizer.mask_token
                masked_string = " ".join(masked_tokens)
                words, probs = self._get_unmasked_words(
                    [masked_string], top_k=500
                )[
                    masked_string
                ]  # get first (only) masked element
                probabilities[:, i] = min(
                    0.5,
                    dict(zip(words, probs)).get(state.example.tokens[i], 0.01),
                )
        elif self.use_constant_probabilities:
            # when generating from an anchor, only keep anchor features fixed (e.g., when doing GA sampling)
            probabilities = np.zeros_like(probabilities)
        else:
            # randomly select each feature (i.e., probability = 0.5)
            probabilities = np.ones_like(probabilities) * 0.5
        # resets candidate anchor features to probability 1 (i.e., always choose)
        probabilities[:, list(candidate.feat_indices)] = 1
        # randomly choose either 0 and 1 for each sample/feature depending on probabilities
        random_sample = np.random.uniform(0, 1, size=probabilities.shape)
        new_data = (random_sample < probabilities).astype(int)
        new_string_data, new_data, new_labels = self.perturb_samples(
            state.example, new_data, state.model
        )
        # update state neighbourhood
        state.neighbourhood.update(new_string_data, new_data, new_labels)
        ## update candidate information
        new_idx = set(
            range(
                state.neighbourhood.current_idx,
                state.neighbourhood.current_idx + n,
            )
        )
        candidate.data_idx = candidate.data_idx.union(
            new_idx
        )  # all new generated samples are covered by candidate
        candidate.num_samples += float(n)
        candidate.num_positives += float(new_labels.sum())
        candidate.precision = candidate.num_positives / candidate.num_samples
        # if candidate in state list, update state candidate info, otherwise, add to state
        state.set_anchor(candidate.feat_indices, candidate)
        return candidate, state
