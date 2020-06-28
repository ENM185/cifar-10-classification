import numpy as np
import math
import cv2 as cv
import pickle
from sklearn.cluster import KMeans

'''
All functions should expect an nx32x32x3 np array
'''

#flatten
def normal_image(data):
    return data.reshape(-1, 32*32*3)

def grayscale(data):
    arr = np.array([0.2989, 0.5870, 0.1140])
    return np.dot(data, arr).reshape(-1, 32*32)

def average(data):
    arr = np.array([1, 1, 1])/float(3)
    return np.dot(data, arr).reshape(-1, 32*32)

def mean_value(data):
    arr = np.array([1, 1, 1]) / 3
    return np.dot(data, arr).reshape(-1, 32*32)

def pixel_histogram(data, window_width=8, bins=16):
    assert 32 % window_width == 0
    num_windows = 32 // window_width
    ret = [] # concatenation of all histograms
    for image in data:
        histograms = []
        for row in range(num_windows):
            for col in range(num_windows):
                for color in range(3):
                    hist = np.histogram(image[row*window_width:(row+1)*window_width,
                                            col*window_width:(col+1)*window_width,
                                            color], bins, (0,1))[0]
                    histograms.extend(np.asarray(hist))
        ret.extend(histograms)
    return np.array(ret).reshape((data.shape[0], -1))

'''
Histogram of oriented gradients
'''
def hog(data, hist_width=8, norm_width=16):
    ret = []
    hist_blocks = int(32 // hist_width)
    blocks_per_norm = int(norm_width // hist_width)
    norm_blocks = hist_blocks-blocks_per_norm+1 # how many configurations there are across a row

    for image in data:
        magnitudes, directions = _gradient(image)
        histograms = np.zeros((hist_blocks,hist_blocks,9)) #square grid

        #create histograms
        for i in range(hist_blocks):
            for j in range(hist_blocks):
                #select proper square from mags and dirs
                values = directions[i*hist_width:(i+1)*hist_width, j*hist_width:(j+1)*hist_width]
                weights = magnitudes[i*hist_width:(i+1)*hist_width, j*hist_width:(j+1)*hist_width]
                histograms[i][j] = np.histogram(values, bins=9, range=(0,180))[0]
        
        #normalize groups of histograms
        for i in range(norm_blocks):
            for j in range(norm_blocks):
                flat_histograms = histograms[i:i+blocks_per_norm, j:j+blocks_per_norm].flatten()
                flat_histograms /= np.linalg.norm(flat_histograms)
                flat_histograms = np.asarray(flat_histograms)
                ret.extend(flat_histograms)

    return np.array(ret).reshape((data.shape[0], -1))

def _gradient(image):
    directions = []
    magnitudes = []
    width = image.shape[0]-2

    # directional derivatives can be found just by shifting and subtracting
    # dx shifts horizontally twice and subtracts from original; dy does the same vertically
    dx = (np.pad(image, ((0,0),(0,2),(0,0)), mode='constant', constant_values=0)
         -np.pad(image, ((0,0),(2,0),(0,0)), mode='constant', constant_values=0))[:,1:-1]
    dy = (np.pad(image, ((0,2),(0,0),(0,0)), mode='constant', constant_values=0)
         -np.pad(image, ((2,0),(0,0),(0,0)), mode='constant', constant_values=0))[1:-1]
    
    magnitudes = np.sqrt(dx**2 + dy**2)
    directions = (np.arctan2(dy, dx) * 180 / np.pi) % 180
    indices = np.argmax(magnitudes, axis=2)
    return np.take_along_axis(magnitudes, np.expand_dims(indices, axis=2), axis=2), np.take_along_axis(directions, np.expand_dims(indices, axis=2), axis=2)

class Sift:
    def __init__(self, kmeans=None):
        self._kmeans = kmeans

    #called when train/test isn't specified
    #if kmeans is already made, default to test
    def sift(self, data):
        if self._kmeans is None:
            return self._calculate_kmeans(data)
        else:
            return self._kmeans_exists(data)

    def _calculate_kmeans(self, data):
        #images = (grayscale(data).reshape((-1,32,32))*255).astype(np.uint8)
        images = (data * 255).astype(np.uint8)
        sift = cv.xfeatures2d.SIFT_create()
        descriptors = []

        # create keypoints on grid (no direction yet)
        keypoints = []
        for i in range(4,32,8):
            for j in range(4,32,8):
                keypoints.append(cv.KeyPoint(i,j,8))
        count = 0
        for img in images:
            magnitudes, directions = _gradient(img)
            idx = 0
            #make directions based on gradient of image
            for i in range(4,32,8):
                for j in range(4,32,8):
                    mags, dirs = magnitudes[i-4:i+4,j-4:j+4].flatten(), directions[i-4:i+4,j-4:j+4].flatten()
                    cutoff = .8 * np.amax(mags)
                    mags = [mag if mag >= cutoff else 0 for mag in mags]
                    keypoints[idx].angle = np.average(dirs, weights=mags) if cutoff != 0 else np.average(dirs)
                    idx += 1

            #sift descriptor; shape: (# of keypoints,128)
            _, desc = sift.compute(img, keypoints)
            descriptors.append(desc)
            count += 1
            if count % 1000 == 0:
                print(count)
        print("Calculating kmeans")
        self._kmeans = KMeans(n_clusters=125,n_jobs=8,precompute_distances=True)
        self._kmeans.fit(np.concatenate(descriptors,axis=0))
        #don't want to have to repeat this work...
        pickle.dump(self._kmeans, open('kmeans', 'wb'))
        ret = []
        for desc in descriptors:
            #create histogram for each image, with a bin for each cluster in KMeans
            ret.append(np.histogram(self._kmeans.predict(desc), range=(0,self._kmeans.n_clusters), bins=self._kmeans.n_clusters)[0])
        
        return np.array(ret)

    def _kmeans_exists(self, data):
        images = (data*255).astype(np.uint8)
        sift = cv.xfeatures2d.SIFT_create()
        ret = []

        keypoints = []
        for i in range(4,32,8):
            for j in range(4,32,8):
                keypoints.append(cv.KeyPoint(i,j,8))

        for img in images:
            magnitudes, directions = _gradient(img)
            idx = 0
            for i in range(4,32,8):
                for j in range(4,32,8):
                    mags, dirs = magnitudes[i-4:i+4,j-4:j+4].flatten(), directions[i-4:i+4,j-4:j+4].flatten()
                    cutoff = .8 * np.amax(mags)
                    mags = [mag if mag >= cutoff else 0 for mag in mags]
                    keypoints[idx].angle = np.average(dirs, weights=mags) if cutoff != 0 else np.average(dirs)
                    idx += 1
            
            _, desc = sift.compute(img, keypoints)

            ret.append(np.histogram(self._kmeans.predict(desc), range=(0,self._kmeans.n_clusters), bins=self._kmeans.n_clusters)[0])

        return np.array(ret)