import numpy as np
import math

'''
All functions should expect an nx32x32x3 np array
'''

#flatten
def normal_image(data):
    return data.reshape(-1, 32*32*3)

def grayscale(data):
    arr = np.array([0.2989, 0.5870, 0.1140])
    return np.dot(data, arr).reshape(-1, 32*32)

def mean_value(data):
    arr = np.array([1, 1, 1]) / 3
    return np.dot(data, arr).reshape(-1, 32*32)

def pixel_histogram(data, window_width=16, bins=16):
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
    #assume image is padded
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