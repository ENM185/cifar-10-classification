from abc import ABC, abstractmethod

class Classifier(ABC):
    def __init__(self):
        self._training_images = None
        self._training_labels = None
        self._test_images = None
        self._test_labels = None

    @abstractmethod
    def train(self, training_images, training_labels):
        pass

    @abstractmethod
    def test(self, test_images, test_labels):
        pass
    