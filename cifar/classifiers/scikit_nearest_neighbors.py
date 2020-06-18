from collections import Counter
import numpy as np
from sklearn.neighbors import NearestNeighbors

from .classifier import Classifier

class ScikitNearestNeighbors(Classifier):
    def __init__(self, k=None, **kw):
        super().__init__(**kw)
        self._k = k
        self._model = NearestNeighbors(p=1) #from scikit module
    
    def train(self, training_images, training_labels):
        self._training_images = training_images
        self._training_labels = training_labels

        self._model.fit(self._training_images)

    def test(self, test_images, test_labels, k=None, verbose=False):
        if k:
            self._k = k
        if self._test_images is not test_images or self._test_labels is not test_labels:
            self._test_images = test_images
            self._test_labels = test_labels
            self._distances = self._model.kneighbors(self._test_images, return_distance=False, n_neighbors=self._k)

        num_correct = 0

        for idx in range(self._distances.shape[0]):
            class_counter = Counter()
            for neighbor in self._distances[idx]:
                class_counter[self._training_labels[neighbor]] += 1
            guess = class_counter.most_common(1)[0][0]

            if guess == self._test_labels[idx]:
                num_correct += 1
        
        proportion_correct = num_correct / self._test_images.shape[0]
        if verbose:
            print("{}/{}, or {}% of the test data was labeled correctly".format(num_correct, self._test_images.shape[0], proportion_correct * 100))
        return proportion_correct
