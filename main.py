import numpy as np
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

train_data = loader.load_training_data("cifar/data")
test_data = loader.load_test_data("cifar/data")
train_data['images'] = np.array(train_data['images']) / float(255)
test_data['images'] = np.array(test_data['images']) / float(255)

#apply filters and choose how many tests to try
test_images = fe.normal_image(test_data['images'])[:30]
test_data = test_data['labels'][:30]

#train and test model
classifier = ScikitNearestNeighbors()
start = time.process_time()
classifier.train(fe.normal_image(train_data['images']), train_data['labels'])
classifier.test(test_images, test_data, k=1)
end = time.process_time()
print("Took {} seconds".format(end - start))

#retest for different k's
results = []
for k in range(1,30):
    results.append([k, classifier.test(test_images, test_data, k=k)])
print(results)
