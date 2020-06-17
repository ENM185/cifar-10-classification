import numpy as np

'''
All functions should expect an nx32x32x3 np array
and will output an mx(32x32x3) np array
'''

#flatten
def normal_image(data):
    return data.reshape(len(data), 32*32*3)

def grayscale(data):
    arr = np.array([0.2989, 0.5870, 0.1140])
    return np.dot(data, arr).reshape(len(data), 32*32)