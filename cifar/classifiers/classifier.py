from abc import ABC, abstractmethod

class Classifier(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def train(self, training_images, training_labels):
        pass

    @abstractmethod
    def test(self, test_images, test_labels):
        pass
    