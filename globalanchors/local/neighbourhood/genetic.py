"""Genetic Algorithm neighbourhood generation for local explainers.

Inspired by LORE (https://arxiv.org/pdf/1805.10820.pdf)."""

from functools import partial
from typing import List, Tuple
from loguru import logger
import numpy as np
from sentence_transformers import util

from globalanchors.local.neighbourhood.base import NeighbourhoodSampler
from globalanchors.anchor_types import (
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
    ):
        self.crossover_prop = (
            crossover_prop  # proportion of population to crossover
        )
        self.mutation_prob = mutation_prob
        self.n_generations = n_generations
        super().__init__(
            use_generator=True,
            use_generator_probabilities=False,
            use_constant_probabilities=True,
        )
        # initialize encoder (generator initialized in base class) for fitness distance
        from sentence_transformers import SentenceTransformer

        self.encoder = SentenceTransformer(
            "all-MiniLM-L6-v2", device=self.device.type
        )

    def _fitness(
        self,
        same_label: bool,
        required_features: List[str],
        example: InputData,
        model: Model,
        genes: List[str],
    ) -> List[float]:
        """Computes fitness for a population."""
        example_text = " ".join(example.tokens)
        if same_label:
            # indicator for matching labels
            label_scores = np.array(
                [
                    1 if model([gene])[0] == example.label else 0
                    for gene in genes
                ]
            )
        else:
            # indicator for mismatched labels
            label_scores = np.array(
                [
                    1 if model([gene])[0] != example.label else 0
                    for gene in genes
                ]
            )
        # distance between embeddings
        gene_embeddings = self.encoder.encode(genes, convert_to_tensor=True)
        example_embedding = self.encoder.encode(
            [example_text], convert_to_tensor=True
        )
        cosine_scores = util.cos_sim(gene_embeddings, example_embedding).cpu()
        dist_scores = np.array(
            [1 - cosine_scores[i][0] for i in range(len(genes))]
        )
        # indicator for identical example
        example_scores = np.array(
            [1 if gene == example_text else 0 for gene in genes]
        )
        # weighted indicator for required features
        features_scores = np.array(
            [
                (
                    np.mean(
                        [
                            1 if feature in gene else 0
                            for feature in required_features
                        ]
                    )
                    if len(required_features) > 0
                    else 1
                )
                for gene in genes
            ]
        )
        return (
            label_scores + dist_scores + features_scores - example_scores
        ).tolist()

    def _select(
        self, population: Population
    ) -> List[Tuple[Individual, Individual]]:
        """Randomly selects a subset of the population to be parents for the next generation proportional to their fitness."""
        n_generated = round(self.crossover_prop * len(population.individuals))
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
        """Randomly replace a random feature for each individual with probability <self.mutation_prob>."""
        mutation_idxs = np.where(
            np.random.uniform(0, 1, len(individuals)) < self.mutation_prob
        )[0]
        masked_strings = []
        masked_ids = []
        # mutate individuals
        for i in mutation_idxs:
            masked_id = np.random.choice(len(individuals[i].gene))
            individuals[i].gene[
                masked_id
            ] = self.fill_masker.tokenizer.mask_token
            masked_strings.append(" ".join(individuals[i].gene))
            masked_ids.append(masked_id)
        results = self._get_unmasked_words(masked_strings)
        # replace mask tokens
        for result_i, data_i in enumerate(mutation_idxs):
            words, probs = results[masked_strings[result_i]]
            masked_word = np.random.choice(words, p=probs)
            individuals[data_i].gene[masked_ids[result_i]] = masked_word
        return individuals

    def _genetic_algorithm(
        self,
        population: Population,
        same_label: bool,
        required_features: List[str],
        example: InputData,
        model: Model,
    ) -> Tuple[np.ndarray, List[str], List[float]]:
        population_size = len(population.individuals)
        data = np.zeros((population_size, len(example.tokens)), dtype=int)
        fitness_fn = partial(
            self._fitness, same_label, required_features, example, model
        )
        for _ in range(self.n_generations):
            # select parents
            parents = self._select(population)
            # crossover parents to create children
            children = self._crossover(parents)
            # mutate children
            new_individuals = self._mutate(population.individuals + children)
            population.individuals = new_individuals
            # evaluate population fitness (and sort + truncate for next generation)
            population.evaluate_population(fitness_fn, population_size)
        # correct data array for matching tokens
        for i, indv in enumerate(population.individuals):
            data[i] = example.tokens == indv.gene
        string_data = [" ".join(indv.gene) for indv in population.individuals]
        fitnesses = [indv.fitness for indv in population.individuals]
        logger.debug(f"Generated strings: {string_data}.")
        return data, string_data, fitnesses

    def generate_samples(
        self,
        example: InputData,
        num_samples: int,
        required_features: List[str],
        model: Model,
    ) -> Tuple[np.ndarray, List[str], List[float]]:
        """_summary_

        Args:
            example (InputData): _description_
            num_samples (int): _description_
            required_features (List[str]): _description_
            model (Model): _description_

        Returns:
            Tuple[np.ndarray, List[str], List[float]]: _description_
        """
        if num_samples == 1:
            # generate a single sample
            population = Population(example, 1)
            data, string_data, fitnesses = self._genetic_algorithm(
                population, True, required_features, example, model
            )
            return data, string_data, fitnesses
        # initialize population from example and evaluate preliminary fitnesses
        different_population = Population(example, num_samples // 2)
        matching_population = Population(
            example, num_samples - num_samples // 2
        )
        # generate samples with matching labels
        matching_data, matching_samples, matching_fitnesses = self._genetic_algorithm(
            matching_population, True, required_features, example, model
        )
        # generate samples with non-matching labels
        different_data, different_samples, different_fitnesses = self._genetic_algorithm(
            different_population, False, required_features, example, model
        )
        # combine samples and return
        return (
            np.vstack((matching_data, different_data)),
            matching_samples + different_samples,
            matching_fitnesses + different_fitnesses
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
        assert len(data) > 0, "Data array must have at least one sample."
        # get anchor required tokens (assuming constant probability => data is the same for all samples)
        required_features = example.tokens[data[0] == 1].tolist()
        data, string_data, _ = self.generate_samples(
            example, data.shape[0], required_features, model
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
