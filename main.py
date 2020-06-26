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

#input feature-extracted TRAINING data
#output will be the weight for f1
def _concat_weights(f1, f2, labels, k=20):
    print("Testing different weights:")
    results = []
    for m in range(10):
        train_data = np.concatenate((m/10*f1[:40000], f2[:40000]), ord='fro')
        train_labels = labels[:45000]
        test_data = np.concatenate((m/10*f1[40000:], f2[40000:]), ord='fro')
        test_labels = labels[45000:]

        classifier = NearestNeighbors()
        classifier.train(train_data, train_labels)
        results.append((classifier.test(test_data, test_labels, k=25), (m/10,1)))

    for m in range(11):
        train_data = np.concatenate((f1[:40000], m/10*f2[:40000]), axis=1)
        train_labels = labels[:45000]
        test_data = np.concatenate((f1[40000:], m/10*f2[40000:]), axis=1)
        test_labels = labels[45000:]

        classifier = NearestNeighbors()
        classifier.train(train_data, train_labels)
        results.append((classifier.test(test_data, test_labels, k=25), (1,m/10)))

    for result in results:
        print(result)

    return max(results)

#features are the arrays of extracted features
def _concat_many_weights(labels, *features):
    accuracy, mult = _concat_weights(features[0], features[1], labels)
    all_mult = mult
    f1 = np.concatenate((mult[0]*features[0], mult[1]*features[1]), axis=1)
    for i in range(2, len(features)):
        accuracy, mult = _concat_weights(f1, features[i], labels)
        f1 = np.concatenate((mult[0]*f1, mult[1]*features[i]), axis=1)
        all_mult += tuple([mult[0] * m for m in all_mult]) + (mult[1],)
    
    return (accuracy, all_mult)

#features are pointers to the functions
def concat(train_images, train_labels, test_images, test_labels, *features):
    train_extracted = []
    norms = []
    for f in features:
        train_extracted.append(f(train_images))
        norms.append(np.linalg.norm(train_extracted[-1], ord='fro'))
        train_extracted[-1] /= norms[-1]
    mult = _concat_many_weights(train_labels, train_extracted)[1]
    print("Final multiplier tuple is {}".format(mult))

    for i in range(len(train_extracted)):
        train_extracted[i] *= mult[i]
    train_data = np.concatenate(tuple(train_extracted), axis=1)
    test_extracted = [mult[i] * features[i](test_images) / norms[i] for i in range(len(features))]
    test_data = np.concatenate(tuple(test_extracted), axis=1)

    classifier = NearestNeighbors()
    classifier.train(train_data, train_labels)
    print("Accuracy on test data:")
    print(classifier.test(test_data, test_labels, k=25))

#load data
train_data = loader.load_training_data("cifar/data")
test_data = loader.load_test_data("cifar/data")

#convert images to floats (0-1)
train_data['images'] = np.array(train_data['images']) / float(255)
test_data['images'] = np.array(test_data['images']) / float(255)

concat(train_data['images'], train_data['labels'], test_data['images'], test_data['labels'], fe.grayscale, fe.hog)

'''
#apply filters and choose how many tests to try
train_grayscale = fe.grayscale(train_data['images'])
norm1 = np.linalg.norm(train_grayscale, ord='fro')
train_grayscale /= norm1
train_hog = fe.hog(train_data['images'])
norm2 = np.linalg.norm(train_hog, ord='fro')
train_hog /= norm2

train_images = np.concatenate((train_grayscale, train_hog), axis=1)
train_labels = train_data['labels']

test_grayscale = fe.grayscale(test_data['images'])
test_grayscale /= norm1
test_hog = fe.hog(test_data['images'])
test_hog /= norm2

test_images = np.concatenate((test_grayscale, test_hog), axis=1)
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
'''