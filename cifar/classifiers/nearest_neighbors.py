from .classifier import Classifier
from scipy.spatial.distance import cdist
from collections import Counter
import numpy as np
import scipy

class NearestNeighbors(Classifier):
    def __init__(self, k=None, **kw):
        super().__init__(**kw)
        self._k = k

    def train(self, training_images, training_labels):
        self._training_images = training_images
        self._training_labels = training_labels

    def test(self, test_images, test_labels, k=None):
        if k:
            self._k = k

        if self._test_images is not test_images or self._test_labels is not test_labels:
            self._test_images = test_images
            self._test_labels = test_labels
            
            # shape: (num_test_images, num_training_images)
            self._distances = cdist(self._test_images, self._training_images, metric='cityblock')

        num_correct = 0

        for ridx in range(self._distances.shape[0]):
            row_distances = [] # (training_img, distance, label) 3-tuples
            row = self._distances[ridx]

            for cidx in range(row.shape[0]):
                row_distances.append((self._training_images[cidx], row[cidx], self._training_labels[cidx]))
            
            row_distances.sort(key=lambda t: t[1]) #sort by distance
            class_counter = Counter()
            for neighbor in row_distances[:self._k]:
                class_counter[neighbor[2]] += 1 # based on class label
            guess = class_counter.most_common(1)[0][0]

            if guess == self._test_labels[ridx]:
                num_correct += 1
        
        proportion_correct = num_correct / self._test_images.shape[0]
        print("{}/{}, or {}% of the test data was labeled correctly".format(num_correct, self._test_images.shape[0], proportion_correct * 100))
        return proportion_correct