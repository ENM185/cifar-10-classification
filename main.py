import numpy as np
import sys
import time
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

#load data
train_data = loader.load_training_data("cifar/data")
test_data = loader.load_test_data("cifar/data")

#convert images to floats (0-1)
train_data['images'] = np.array(train_data['images']) / float(255)
test_data['images'] = np.array(test_data['images']) / float(255)

#apply filters and choose how many tests to try
train_images = fe.pixel_histogram(train_data['images'])
train_labels = train_data['labels']
test_images = fe.pixel_histogram(test_data['images'])
test_labels = test_data['labels']

#train and test model
print("Training and testing model")
classifier = NearestNeighbors()
start = time.process_time()
classifier.train(train_images, train_labels)
classifier.test(test_images, test_labels, k=1)
end = time.process_time()
print("Training and testing took {} seconds".format(end - start))

#retest for different k's
#doesn't work with scikit
results = []
for k in range(1,30):
    results.append([k, classifier.test(test_images, test_labels, k=k)])
for row in results:
    print("({},{})".format(row[0],row[1]))
