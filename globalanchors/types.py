"""Types used in the library."""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Set,
    Tuple,
    TypedDict,
    Union,
)
import numpy as np
import spacy

from globalanchors.utils import normalize_feature_indices, exp_normalize

from loguru import logger

## Training Types

Model = Callable[[List[str]], List[int]]


class Dataset(TypedDict):
    train_data: List[str]
    train_labels: List[int]
    val_data: List[str]
    val_labels: List[int]
    test_data: List[str]
    test_labels: List[int]


class LocalMetrics(TypedDict):
    rule_length: float
    coverage: float
    precision: float
    f1: float
    num_samples: float
    time_taken: float


class GlobalMetrics(TypedDict):
    # statistics for generated global ruleset
    global_rule_length: float
    global_rule_coverage: float
    global_rule_precision: float
    global_rule_f1: float
    # statistics for predicting on dataset
    average_valid_rules: float
    rule_length: float
    accuracy: float


## Anchors Types


@dataclass
class InputData:
    text: str
    tokens: np.array  # tokenized text (n_features)
    positions: np.array  # positions for tokenized text (n_features)
    label: int

    def __init__(self, text: str, model: Model):
        self.nlp = spacy.load("en_core_web_sm")
        self.text = text
        processed = self.nlp(text)
        self.tokens = np.array([x.text for x in processed], dtype="|U80")
        self.positions = np.array([x.idx for x in processed])
        self.label = model([text])[0]


@dataclass
class NeighbourhoodData:
    string_data: np.ndarray  # input string for each sample (n x 1)
    data: np.ndarray  # index toggles for each sample (n x n_features)
    labels: np.array  # model output for each sample (n)
    current_idx: int
    prealloc_size: int

    def __init__(
        self, string_data: np.ndarray, data: np.ndarray, labels: np.array
    ):
        self.string_data = string_data
        self.data = data
        self.labels = labels
        self.current_idx = self.data.shape[0]
        self.prealloc_size = 10000

    def reallocate(self):
        """Pre-allocate new memory to internal storage arrays."""
        self.string_data = np.vstack(
            (
                self.string_data,
                np.zeros(
                    (self.prealloc_size, self.string_data.shape[1]),
                    self.string_data.dtype,
                ),
            )
        )
        self.data = np.vstack(
            (
                self.data,
                np.zeros(
                    (self.prealloc_size, self.data.shape[1]), self.data.dtype
                ),
            )
        )
        self.labels = np.hstack(
            (self.labels, np.zeros(self.prealloc_size, self.labels.dtype))
        )

    def update(
        self,
        new_string_data: np.ndarray,
        new_data: np.ndarray,
        new_labels: np.array,
    ):
        """Update existing neighbourhood data with new data."""
        # update existing string_data dtype character lengths
        if "<U" in str(new_string_data.dtype):
            # set max string dtype to avoid string truncation (e.g., '<U308', '<U290' -> '<U308')
            max_dtype = max(
                str(self.string_data.dtype), str(new_string_data.dtype)
            )
            self.string_data = self.string_data.astype(max_dtype)
            new_string_data = new_string_data.astype(max_dtype)
        ## allocating necessary memory
        new_idx = range(self.current_idx, self.current_idx + new_data.shape[0])
        # make sure pre-allocated data arrays have enough space
        while self.data.shape[0] < self.current_idx + new_data.shape[0]:
            self.reallocate()
        # update neighbourhood arrays
        self.string_data[new_idx] = new_string_data
        self.data[new_idx] = new_data
        self.labels[new_idx] = new_labels
        self.current_idx += new_data.shape[0]


@dataclass
class CandidateAnchor:
    feats: Set[str] = field(default_factory=set)
    feat_indices: Set[int] = field(default_factory=set)
    precision: float = 0
    coverage: float = 0
    prediction: int = 1
    num_samples: int = 0
    num_positives: int = 0
    data_idx: Set[int] = field(default_factory=set)
    coverage_idx: Set[int] = field(default_factory=set)


@dataclass
class BeamState:
    neighbourhood: NeighbourhoodData
    anchors: Dict[Tuple[int], CandidateAnchor]
    coverage_data: (
        np.ndarray
    )  # fixed data to calculate coverage (n_c x n_features)
    n_features: int
    example: InputData
    model: Model

    def get_anchor(self, feature_indices: Iterable[int]) -> CandidateAnchor:
        """Finds an anchor in self.anchors based on feature indices in any order."""
        normalized_indices = normalize_feature_indices(feature_indices)
        if normalized_indices not in self.anchors:
            logger.warning(
                f"No matching anchors found for query: {feature_indices}. Returning None."
            )
            return None
        return self.anchors[normalized_indices]

    def set_anchor(
        self, feature_indices: Iterable[int], anchor: CandidateAnchor
    ) -> None:
        normalized_indices = normalize_feature_indices(feature_indices)
        self.anchors[normalized_indices] = anchor

    def initialize_features(self):
        """Initializes candidates for individual features based on the neighbourhood."""
        data = self.neighbourhood.data[: self.neighbourhood.current_idx]
        labels = self.neighbourhood.labels[: self.neighbourhood.current_idx]
        feature_indices = range(self.n_features)
        for f_i in feature_indices:
            # calculate data indices covered by candidate anchor
            feat_present = data[:, f_i].nonzero()[0]
            # calculate data indices and precision
            covered_indices = set(feat_present)
            num_samples = float(len(covered_indices))
            num_positives = float(labels[list(covered_indices)].sum())
            precision = num_positives / num_samples if num_samples > 0 else 0
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
            new_anchor = CandidateAnchor(
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
            self.set_anchor([f_i], new_anchor)


class ExplainerOutput(TypedDict):
    example: str
    explanation: Set[str]  # features used by anchor
    precision: float
    coverage: float
    prediction: int
    num_samples: int


class GlobalExplainerOutput(TypedDict):
    example: str
    explanations: List[ExplainerOutput]
    rule_used: Union[ExplainerOutput | None]
    prediction: int


## Genetic Algorithm Types

DistanceFunctionType = Literal["cosine", "neuclid"]


@dataclass
class Individual:
    gene: np.array
    fitness: float

    def __init__(self, gene: np.array, fitness: float):
        self.gene = gene
        self.fitness = fitness

    def __lt__(self, other):
        return self.fitness < other.fitness

    def __le__(self, other):
        return self.fitness <= other.fitness

    def __eq__(self, other):
        return self.fitness == other.fitness

    def __ne__(self, other):
        return self.fitness != other.fitness

    def __gt__(self, other):
        return self.fitness > other.fitness

    def __ge__(self, other):
        return self.fitness >= other.fitness

    def copy(self) -> "Individual":
        return deepcopy(self)


@dataclass
class Population:
    individuals: List[Individual]
    probs: List[float]

    def __init__(
        self,
        example: InputData,
        num_samples: int,
    ):
        # setup initial population
        initial_fitness = 0
        self.individuals = sorted(
            [
                Individual(example.tokens, initial_fitness)
                for _ in range(num_samples)
            ]
        )
        self.probs = (np.ones(num_samples) / num_samples).tolist()

    def evaluate_population(
        self, fitness_fn: Callable[[Individual], float], population_size: int
    ):
        """Calculates fitness for each individual, sorts the population, and normalizes fitness values into probabilities."""
        for indv in self.individuals:
            indv.fitness = fitness_fn(indv.gene)
        # sort individuals
        self.individuals = sorted(self.individuals, key=lambda x: x.fitness)
        # truncate population
        self.individuals = self.individuals[:population_size]
        fitnesses = [indv.fitness for indv in self.individuals]
        # calculate normalized probabilities
        if sum(fitnesses) == 0:  # handle 0 prob case
            self.probs = (np.ones(len(fitnesses)) / len(fitnesses)).tolist()
        else:
            self.probs = exp_normalize(
                np.array(fitnesses) / sum(fitnesses)
            ).tolist()
