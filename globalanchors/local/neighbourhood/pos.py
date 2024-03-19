"""Part-of-Speech replacement neighbourhood generation for local explainers.

Used by the original Anchors paper (https://ojs.aaai.org/index.php/AAAI/article/view/11491)."""

from typing import Tuple, override
import numpy as np

from globalanchors.local.neighbourhood.base import NeighbourhoodSampler
from globalanchors.types import InputData, Model


class PartOfSpeechSampler(NeighbourhoodSampler):
    def __init__(
        self, use_generator_probabilities: bool = False, one_pass: bool = False
    ):
        self.one_pass = one_pass
        super().__init__(
            use_generator=True,
            use_generator_probabilities=use_generator_probabilities,
        )

    @override
    def perturb_samples(
        self,
        example: InputData,
        data: np.ndarray,
        model: Model,
        compute_labels: bool,
    ) -> Tuple[np.ndarray, np.ndarray, np.array]:
        """Generates new text strings and labels in the neighbourhood of an example given the words in the example to replace."""
        # create array of string tokens
        raw_data = np.zeros(data.shape, "|U80")
        raw_data[:] = example.tokens
        # set all disabled tokens to a mask token
        raw_data[data == 0] = self.bert_tokenizer.mask_token
        # concatenate tokens to get masked string data
        masked_data = [" ".join(string_array) for string_array in raw_data]
        # replace mask tokens with generated words
        if self.one_pass:  # replace all at once
            for i, masked_string in enumerate(masked_data):
                results = self._get_unmasked_words(masked_string)
                masked_words = np.array(
                    [
                        np.random.choice(words, p=probs)
                        for words, probs in results
                    ]
                )
                # replace mask tokens
                raw_data[i, data[i] == 0] = masked_words
                # correct data array for matching tokens
                data[i] = example.tokens == raw_data[i]
        else:  # replace one by one
            for i, masked_string in enumerate(masked_data):
                for j in np.where(data[i] == 0)[0]:
                    words, probs = self._get_unmasked_words(masked_string)[j]
                    masked_word = np.random.choice(words, p=probs)
                    # replace mask tokens
                    raw_data[i, j] = masked_word
                # correct data array for matching tokens
                data[i] = example.tokens == raw_data[i]
        string_data = [" ".join(string_array) for string_array in raw_data]
        # compute labels (optional)
        labels = np.ones(data.shape[0], dtype=int)
        if compute_labels:
            labels = np.array(model(string_data)).astype(int)
        # convert string list into numpy array
        max_string_len = max([len(string) for string in string_data])
        string_dtype = f"|U{max(80, max_string_len)}"
        string_data = np.array(string_data, dtype=string_dtype).reshape(
            -1, 1
        )  # column array
        return string_data, data, labels
