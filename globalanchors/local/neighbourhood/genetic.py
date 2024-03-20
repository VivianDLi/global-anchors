"""Genetic Algorithm neighbourhood generation for local explainers.

Inspired by LORE (https://arxiv.org/pdf/1805.10820.pdf)."""

from functools import partial
from typing import List, Tuple, override
import numpy as np
import torch

from globalanchors.local.neighbourhood.base import NeighbourhoodSampler
from globalanchors.local.utils import get_distance_function
from globalanchors.types import (
    InputData,
    Model,
    Individual,
    Population,
    DistanceFunctionType,
)


class GeneticAlgorithmSampler(NeighbourhoodSampler):
    def __init__(
        self,
        crossover_prop: float = 0.5,
        mutation_prob: float = 0.2,
        n_generations: int = 10,
        distance_function: DistanceFunctionType = "neuclid",
    ):
        self.crossover_prop = (
            crossover_prop  # proportion of population to crossover
        )
        self.mutation_prob = mutation_prob
        self.n_generations = n_generations
        self.distance_function = get_distance_function(distance_function)
        super().__init__(
            use_generator=True,
            use_generator_probabilities=False,
            use_constant_probabilities=True,
        )
        # initialize encoder (generator initialized in base class) for fitness distance
        from transformers import DistilBertModel

        self.encoder = DistilBertModel.from_pretrained("distilbert-base-cased")
        self.encoder.to(self.device)
        self.encoder.eval()
        self.encoder_cache = {}

    def _encode(self, text: str) -> np.array:
        """Encodes a string into a numerical representation."""
        # check for cached output for consistency + optimization
        if text in self.encoder_cache:
            return self.encoder_cache[text]
        encoded_input = torch.tensor(
            self.bert_tokenizer.encode(text, add_special_tokens=True)
        )
        model_input = torch.tensor([encoded_input], device=self.device)
        with torch.no_grad():  # for memory optimization
            encoding = self.bert(model_input)[0][0, 0, :].numpy()
        self.encoder_cache[text] = encoding
        return self.encoder_cache[text]

    def _fitness(
        self,
        same_label: bool,
        required_features: List[str],
        example: InputData,
        model: Model,
        gene: np.array,
    ) -> float:
        indv_text = " ".join(gene)
        example_text = " ".join(example.tokens)
        if same_label:
            # indicator for matching labels
            label_score = 1 if model([indv_text])[0] == example.label else 0
        else:
            # indicator for mismatched labels
            label_score = 1 if model([indv_text])[0] != example.label else 0
        # distance between embeddings
        dist_score = self.distance_function(
            self._encode(indv_text), self._encode(example_text)
        )
        # indicator for identical example
        example_score = 1 if indv_text == example_text else 0
        # weighted indicator for required features
        features_score = np.mean(
            [1 if feature in indv_text else 0 for feature in required_features]
        )
        return label_score + (1 - dist_score) + features_score - example_score

    def _select(
        self, population: Population
    ) -> List[Tuple[Individual, Individual]]:
        """Randomly selects a subset of the population to be parents for the next generation proportional to their fitness."""
        n_generated = self.crossover_prop * len(population.individuals)
        parents = []
        for _ in range(n_generated):
            # select two parents at random
            parent1, parent2 = np.random.choice(
                population.individuals, 2, p=population.probs, replace=False
            )
            parents.append((parent1, parent2))
        return parents

    def _crossover(
        self, parents: List[Tuple[Individual, Individual]]
    ) -> List[Individual]:
        """Generates children by combining genes of two random parents.
        Combines genes by randomly selecting two crossover points and swapping genes in the interval between.
        """
        children = []
        for parent1, parent2 in parents:
            gene1, gene2 = parent1.gene, parent2.gene
            # select crossover points
            crossover_points = np.random.choice(len(gene1), 2, replace=False)
            crossover_points.sort()
            # swap genes between crossover points
            new_gene = gene1.copy()
            new_gene[crossover_points[0] : crossover_points[1]] = gene2[
                crossover_points[0] : crossover_points[1]
            ]
            children.append(Individual(new_gene, 0))
        return children

    def _mutate(self, individuals: List[Individual]) -> List[Individual]:
        """For each gene in each individual, randomly replace with a new gene with probability <self.mutation_prob>."""
        for indv in individuals:
            masked_idx = (
                np.random.uniform(0, 1, len(indv.gene)) < self.mutation_prob
            )
            indv.gene[masked_idx] = self.bert_tokenizer.mask_token
            masked_string = " ".join(indv.gene)
            results = self._get_unmasked_words(masked_string)
            masked_words = np.array(
                [np.random.choice(words, p=probs) for words, probs in results]
            )
            # replace mask tokens
            indv.gene[masked_idx] = masked_words
        return individuals

    def _genetic_algorithm(
        self,
        population: Population,
        same_label: bool,
        required_features: List[str],
        example: InputData,
        model: Model,
    ) -> Tuple[np.ndarray, List[str]]:
        population_size = len(population.individuals)
        data = np.zeros((population_size, len(example.tokens)), dtype=int)
        fitness_fn = partial(
            self._fitness, same_label, required_features, example, model
        )
        for _ in self.n_generations:
            # select parents
            parents = self._select(population)
            # crossover parents to create children
            children = self._crossover(parents)
            # mutate children
            new_individuals = self._mutate(population.individuals + children)
            population.individuals.extend(new_individuals)
            # evaluate population fitness (and sort)
            population.evaluate_fitness(fitness_fn)
            # update (select the fittest individuals)
            population.individuals = population.individuals[:population_size]
        # correct data array for matching tokens
        for i, indv in population.indiviuals:
            data[i] = example.tokens == indv.gene
        return data, [" ".join(indv.gene) for indv in population.individuals]

    def generate_samples(
        self,
        example: InputData,
        num_samples: int,
        required_features: List[str],
        model: Model,
    ) -> Tuple[np.ndarray, List[str]]:
        """_summary_

        Args:
            example (InputData): _description_
            num_samples (int): _description_
            required_features (List[str]): _description_
            model (Model): _description_

        Returns:
            Tuple[np.ndarray, List[str]]: _description_
        """
        # initialize population from example and evaluate preliminary fitnesses
        initial_population = Population(example, num_samples / 2)
        # generate samples with matching labels
        matching_data, matching_samples = self._genetic_algorithm(
            initial_population.copy(), True, required_features, example, model
        )
        # generate samples with non-matching labels
        different_data, different_samples = self._genetic_algorithm(
            initial_population.copy(), False, required_features, example, model
        )
        # combine samples and return
        return (
            np.vstack((matching_data, different_data)),
            matching_samples + different_samples,
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
        assert len(data) > 0, "Data array must have at least one sample."
        # get anchor required tokens (assuming constant probability => data is the same for all samples)
        required_features = example.tokens[data[0] == 1].tolist()
        data, string_data = self.generate_samples(
            example, data.shape[0], required_features
        )
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
