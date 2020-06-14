def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        d = pickle.load(fo, encoding='bytes')
    return d


def reshape_images(images):
    return images.reshape(-1, 3, 32, 32).transpose([0, 2, 3, 1])


def load_training_data(path_to_dataset):
    import os
    import numpy
    
    training_data = {"images": [], "labels": [], "filenames": []}
    
    for i in range(1,6):
        batch = unpickle(
            os.path.join(path_to_dataset, "data_batch_{}".format(i)))
        
        training_data["images"].append(batch[b"data"])
        training_data["labels"].extend(batch[b"labels"])
        training_data["filenames"].extend(map(bytes.decode, batch[b"filenames"]))
    
    training_data["images"] = numpy.concatenate(training_data["images"], axis=0)
    training_data["images"] = reshape_images(training_data["images"])
    return training_data


def load_test_data(path_to_dataset):
    import os
    
    batch = unpickle(os.path.join(path_to_dataset, "test_batch"))
    
    test_data = {}
    test_data["images"] = batch[b"data"]
    test_data["labels"] = batch[b"labels"]
    test_data["filenames"] = list(map(bytes.decode, batch[b"filenames"]))
    
    test_data["images"] = reshape_images(test_data["images"])
    return test_data

label_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
