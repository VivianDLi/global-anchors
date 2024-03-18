"""Types used in the library."""

from dataclasses import dataclass
from typing import List, Literal, Set, TypedDict
import numpy as np
import spacy

from globalanchors.utils import normalize_feature_indices

from loguru import logger

NeighbourhoodSamplerType = Literal["GA", "POS", "UNK"]
ModelType = Literal["SVM", "RF", "MLP"]


@dataclass
class InputData:
    text: str
    tokens: np.array
    positions: np.array
    label: int

    def __init__(self, text, model):
        self.nlp = spacy.load("en_core_web_sm")
        self.text = text
        processed = self.nlp(text)
        self.tokens = np.array([x.text for x in processed], dtype="|U80")
        self.positions = np.array([x.idx for x in processed])
        # TODO: text definitely needs to have some preprocessing I think, but this is for multi-class support
        self.label = model(text)


@dataclass
class NeighbourhoodData:
    raw_data: np.ndarray  # words for each sample
    data: np.ndarray  # index toggles for each sample
    labels: np.array  # model output for each sample
    current_idx: int
    prealloc_size: int

    def reallocate(self):
        """Pre-allocate new memory to internal storage arrays."""
        self.raw_data = np.vstack(
            self.raw_data,
            np.zeros(
                (self.prealloc_size, self.raw_data.shape[1]),
                self.raw_data.dtype,
            ),
        )
        self.data = np.vstack(
            self.data,
            np.zeros(
                (self.prealloc_size, self.data.shape[1]), self.data.dtype
            ),
        )
        self.labels = np.hstack(
            (self.labels, np.zeros(self.prealloc_size, self.labels.dtype))
        )


@dataclass
class CandidateAnchor:
    feats: Set[str] = set()
    feat_indices: Set[int] = set()
    precision: float = 0
    coverage: float = 0
    prediction: int = 1
    num_samples: int = 0
    num_positives: int = 0
    data_idx: Set[int] = set()
    coverage_idx: Set[int] = set()


@dataclass
class BeamState:
    neighbourhood: NeighbourhoodData
    anchors: List[CandidateAnchor]
    coverage_data: np.ndarray  # fixed data to calculate coverage
    n_features: int
    example: InputData

    def find_anchor(self, feature_indices):
        """Finds an anchor in self.anchors based on feature indices in any order."""
        normalized_indices = normalize_feature_indices(feature_indices)
        matches = [
            anchor
            for anchor in self.anchors
            if normalize_feature_indices(anchor.feat_indices)
            == normalized_indices
        ]
        if len(matches) == 0:
            logger.warning(
                f"No matching anchors found for query: {feature_indices}. Returning None."
            )
            return None
        return matches[0]

    def initialize_features(self):
        """Initializes candidates for individual features based on the neighbourhood."""
        data = self.neighbourhood.data[: self.neighbourhood.current_idx]
        labels = self.neighbourhood.labels[: self.neighbourhood.current_idx]
        feature_indices = range(self.n_features)
        new_candidates = []
        for f_i in feature_indices:
            # calculate data indices covered by candidate anchor
            feat_present = data[:, f_i].nonzero()[0]
            # calculate data indices and precision
            covered_indices = set(feat_present)
            num_samples = float(len(covered_indices))
            num_positives = float(labels[covered_indices].sum())
            precision = num_positives / num_samples
            # calculate coverage indices and coverage
            coverage_present = self.coverage_data[:, f_i].nonzero()[0]
            coverage_indices = set(coverage_present)
            coverage = (
                float(len(coverage_indices)) / self.coverage_data.shape[0]
            )
            # define metadata information
            prediction = self.example.label
            feats = set([self.example.tokens[f_i]])
            feat_indices = set([f_i])
            new_candidates.append(
                CandidateAnchor(
                    feats,
                    feat_indices,
                    precision,
                    coverage,
                    prediction,
                    num_samples,
                    num_positives,
                    covered_indices,
                    coverage_indices,
                )
            )
        self.anchors = new_candidates


class ExplainerOutput(TypedDict):
    example: str
    explanation: Set[str]
    precision: float
    coverage: float
    prediction: int
