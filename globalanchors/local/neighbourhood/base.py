"""Base general implementation of neighbourhood generation for local explainers."""

from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np
import torch

from loguru import logger

from globalanchors.local.utils import exp_normalize
from globalanchors.types import (
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
        if self.use_constant_probabilities:
            self.use_generator = False
            self.use_generator_probabilities = False
            logger.info(
                "For constant probabilities, the generator will not be initialized. Setting <use_generator> to False."
            )
        if self.use_generator_probabilities:
            self.use_generator = True
            logger.info(
                "For generator probabilities to be used, generator must be enabled. Setting <use_generator> to True."
            )
        if self.use_generator:
            from transformers import DistilBertTokenizer, DistilBertForMaskedLM

            # import BERT for text generation
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            self.bert_tokenizer = DistilBertTokenizer.from_pretrained(
                "distilbert-base-cased"
            )
            self.bert = DistilBertForMaskedLM.from_pretrained(
                "distilbert-base-cased"
            )
            self.bert.to(self.device)
            self.bert.eval()
            self.bert_cache = {}

    def _get_unmasked_words(
        self, masked_text: str
    ) -> List[Tuple[List[str], List[float]]]:
        """Gets a list of generated words and probabilities for each word for each masked token."""
        assert (
            self.use_generator
        ), "Text generator must be enabled to generate unmasked words."
        # check for cached output for consistency + optimization
        if masked_text in self.bert_cache:
            return self.bert_cache[masked_text]
        encoded_input = torch.tensor(
            self.bert_tokenizer.encode(masked_text, add_special_tokens=True)
        )
        masked_inputs = (
            (encoded_input == self.bert_tokenizer.mask_token_id)
            .numpy()
            .nonzero()[0]
        )
        model_input = torch.tensor([encoded_input], device=self.device)
        with torch.no_grad():  # for memory optimization
            output = self.bert(model_input)[0]
        # extract words and probabilities from model output
        generated_words = []
        for i in masked_inputs:
            probs, top_words = torch.topk(
                output[0, i], 500
            )  # top 500 generated words
            words = self.bert_tokenizer.convert_ids_to_tokens(top_words)
            probs = np.array([float(prob) for prob in probs])
            generated_words.append((words, probs))
        self.bert_cache[masked_text] = [
            (words, exp_normalize(probs)) for words, probs in generated_words
        ]
        return self.bert_cache[masked_text]

    @abstractmethod
    def perturb_samples(
        self,
        example: InputData,
        data: np.ndarray,
        model: Model,
        compute_labels: bool,
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
    ) -> NeighbourhoodData:
        """Samples data from the neighbourhood of an example, optionally updating a given neighbourhood.

        Args:
            example (InputData): Example to generate samples around.
            model (_type_): Model to explain and use to predict labels.
            n (int, optional): Number of new examples to generate. Defaults to 1.
            neighbourhood (NeighbourhoodData, optional): Optional neighbourhood to add newly generated examples to. Defaults to None.
            compute_labels (bool, optional): Whether to also compute model labels for samples or not. Defaults to True.

        Returns:
            NeighbourhoodData: NeighbourhoodData object representing generated samples in the example neighbourhood.
        """
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
                masked_tokens[i] = self.bert_tokenizer.mask_token
                masked_string = " ".join(masked_tokens)
                words, probs = self._get_unmasked_words(masked_string)[
                    0
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
            Tuple[CandidateAnchor, BeamState]: _description_
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
                masked_tokens[i] = self.bert_tokenizer.mask_token
                masked_string = " ".join(masked_tokens)
                words, probs = self._get_unmasked_words(masked_string)[
                    0
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
        probabilities[:, candidate.feat_indices] = 1
        # randomly choose either 0 and 1 for each sample/feature depending on probabilities
        random_sample = np.random.uniform(0, 1, size=probabilities.shape)
        new_data = (random_sample < probabilities).astype(int)
        new_string_data, new_data, new_labels = self.perturb_samples(
            state.example, new_data, state.model
        )
        # update state neighbourhood
        state.neighbourhood.update(new_string_data, new_data, new_labels)
        ## update candidate information
        new_idx = set(range(self.current_idx, self.current_idx + n))
        candidate.data_idx = candidate.data_idx.union(
            new_idx
        )  # all new generated samples are covered by candidate
        candidate.num_samples += float(n)
        candidate.num_positives += float(new_labels.sum())
        candidate.precision = candidate.num_positives / candidate.num_samples
        # if candidate in state list, update state candidate info, otherwise, add to state
        state.set_anchor(candidate.feat_indices, candidate)
        return candidate, state
