from abc import ABC, abstractmethod

class Classifier(ABC):
    def __init__(self, training_images, training_labels, test_images, test_labels):
        self._training_images = training_images
        self._training_labels = training_labels
        self._test_images = test_images
        self._test_labels = test_labels

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def test(self):
        pass
    