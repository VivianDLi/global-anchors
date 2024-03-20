"""Implementations of SVM, RF, and NN using scikit-learn."""

from abc import ABC
from typing import List

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from globalanchors.types import Dataset


class BaseModel(ABC):
    def __init__(self, model):
        self.vectorizer = CountVectorizer(min_df=1)
        self.model = model

    def train(self, dataset: Dataset):
        train_vectors = self.vectorizer.fit_transform(
            np.array(dataset.train_data)
        )
        train_labels = np.array(dataset.train_labels)
        self.model.fit(train_vectors, train_labels)

    def __call__(self, strings: List[str]) -> List[int]:
        vectors = self.vectorizer.transform(np.array(strings))
        return self.model.predict(vectors)


class SVM(BaseModel):
    def __init__(self):
        super().__init__(SVC())


class RF(BaseModel):
    def __init__(self):
        super().__init__(RandomForestClassifier())


class NN(BaseModel):
    def __init__(self):
        super().__init__(MLPClassifier())
