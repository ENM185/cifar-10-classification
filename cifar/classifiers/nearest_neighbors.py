from .classifier import Classifier
from scipy.spatial.distance import cdist
from collections import Counter
import numpy as np
import scipy

class NearestNeighbors(Classifier):
    def __init__(self, k, **kw):
        super().__init__(**kw)
        self._k = k

    def train(self):
        pass

    def test(self):
        num_correct = 0

        distances = cdist(self._test_images, self._training_images, metric='minkowski',p=1)

        row_distances = [] # (training_img, distance, label) 3-tuples
        for ridx in range(len(distances)):
            row = distances[ridx]
            for cidx in range(len(row)):
                row_distances.append((self._training_images[cidx], row[cidx], self._training_labels[cidx]))
            
            row_distances.sort(key=lambda t: t[1]) #sort by distance
            class_counter = Counter()
            for neighbor in row_distances[:self._k]:
                class_counter[neighbor[2]] += 1 # based on class label
            guess = class_counter.most_common(1)[0][0]

            print("{} {}".format(guess, self._test_labels[ridx]))
            if guess == self._test_labels[ridx]:
                num_correct += 1
        
        proportion_correct = num_correct / len(self._test_images)
        print("{}/{}, or {}% of the test data was labeled correctly".format(num_correct, len(self._test_images), proportion_correct * 100))
        return proportion_correct