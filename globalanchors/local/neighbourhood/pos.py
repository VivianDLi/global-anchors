"""Part-of-Speech replacement neighbourhood generation for local explainers.

Used by the original Anchors paper (https://ojs.aaai.org/index.php/AAAI/article/view/11491)."""

from typing import Tuple
import numpy as np

from globalanchors.local.neighbourhood.base import NeighbourhoodSampler
from globalanchors.anchor_types import InputData, Model


class PartOfSpeechSampler(NeighbourhoodSampler):
    def __init__(self, use_generator_probabilities: bool = False):
        super().__init__(
            use_generator=True,
            use_generator_probabilities=use_generator_probabilities,
        )

    # override
    def perturb_samples(
        self,
        example: InputData,
        data: np.ndarray,
        model: Model,
        compute_labels: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.array]:
        """Generates new text strings and labels in the neighbourhood of an example given the words in the example to replace."""
        # create array of string tokens
        raw_data = np.zeros(data.shape, "|U80")
        raw_data[:] = example.tokens
        # replace mask tokens with generated words one by one
        for i, data_row in enumerate(data):
            for string_i in np.where(data_row == 0)[0]:
                raw_data[i, string_i] = self.fill_masker.tokenizer.mask_token
                masked_string = " ".join(raw_data[i])
                words, probs = self._get_unmasked_words([masked_string])[
                    masked_string
                ]
                masked_word = np.random.choice(words, p=probs)
                # replace mask tokens
                raw_data[i, string_i] = masked_word
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
