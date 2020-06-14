import cifar.load_data as loader
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def show_image(index):
    data = np.array(train_data['images'][index])
    img = Image.fromarray(data)

    plt.imshow(data)
    plt.title("Image {}: {}".format(index, loader.label_names[train_data['labels'][index]]))
    plt.show()

train_data = loader.load_training_data("cifar/data")
test_data = loader.load_test_data("cifar/data")
print(train_data['filenames'][0 ])
show_image(0)


