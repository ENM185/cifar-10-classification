import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

from cifar import feature_extraction as fe
from cifar import load_data as loader
from cifar.classifiers.classifier import Classifier
from cifar.classifiers.nearest_neighbors import NearestNeighbors
from cifar.classifiers.scikit_nearest_neighbors import ScikitNearestNeighbors

def show_image(set, index):
    data = set['images'][index]
    img = Image.fromarray(data)

    plt.imshow(data)
    #plt.title("Image {}: {}".format(index, loader.label_names[set['labels'][index]]))
    plt.show()

train_data = loader.load_training_data("cifar/data")
test_data = loader.load_test_data("cifar/data")
train_data['images'] = np.array(train_data['images'])
test_data['images'] = np.array(test_data['images'])


classifier = NearestNeighbors(15, training_images=fe.grayscale(train_data['images']), training_labels=train_data['labels'], test_images=fe.grayscale(test_data['images'][0:10]), test_labels=test_data['labels'][0:10])

classifier.train()
classifier.test()
