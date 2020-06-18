from collections import Counter
import numpy as np
from sklearn.neighbors import NearestNeighbors

from .classifier import Classifier

class ScikitNearestNeighbors(Classifier):
    def __init__(self, k, **kw):
        super().__init__(**kw)
        self._model = NearestNeighbors(n_neighbors=k, p=1) #from scikit module
    
    def train(self):
        self._model.fit(self._training_images)

    def test(self):
        num_correct = 0
        distances = self._model.kneighbors(self._test_images, return_distance=False)

        for idx in range(len(distances)):
            class_counter = Counter()
            for neighbor in distances[idx]:
                class_counter[self._training_labels[neighbor]] += 1
            guess = class_counter.most_common(1)[0][0]

            #print("{} {}".format(guess, self._test_labels[idx]))
            if guess == self._test_labels[idx]:
                num_correct += 1
        
        proportion_correct = num_correct / len(self._test_images)
        print("{}/{}, or {}% of the test data was labeled correctly".format(num_correct, len(self._test_images), proportion_correct * 100))
        return proportion_correct
