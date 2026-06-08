import numpy as np
import struct
from array import array
from os.path  import join
import pickle
'''
    File:       MNIST_fileHandler.py 
    (TLDR)

    Purpose:    Loading, unpacking and writing the MNIST dataset

    Use:        Calling 'pkl_write()' writes the flattened normalized data into files.
                Calling 'pkl_read()' reads the files made by 'pkl_write()' so it needs to have been called once

'''


# MNIST Data Loader Class from (https://www.kaggle.com/code/hojjatk/read-mnist-dataset)
class MnistDataloader(object):
    def __init__(self, training_images_filepath,training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(self, images_filepath, labels_filepath):        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img            
        
        return images, labels
            
    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train),(x_test, y_test)        
    



# Loading the files (called in 'write_as_flat()')
def load_MNIST_files():
    input_path = 'MNIST_files'
    training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
    training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
    test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
    test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

    mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
    (images_train, labels_train), (images_test, labels_test) = mnist_dataloader.load_data()

    #checking and setting sizes
    size_training = len(images_train) #number of training examples
    assert size_training == len(labels_train), "label/image size mismatch in 'training' data"

    size_test     = len(images_test) #number of test examples
    assert size_test == len(labels_test), "label/image size mismatch in 'test' data"

    width_image    =len(images_train[0][0]) #width of image, assumed square
    assert width_image == len(images_train[0]), "non-square data in set"

    return (images_train, labels_train), (images_test, labels_test), (size_training, size_test, width_image)




# Writing as tuples (arr,label),
#       arr as flattened array normalized to [0,1],
#       label as label for arr,
#               into .pkl file
def pkl_write():
    (images_train, labels_train), (images_test, labels_test), (size_training, size_test, width_image)  = load_MNIST_files()

    #write training data
    flat_data_train = []
    max = 0
    for i in range(size_training):
        flat_data_train.append((np.concatenate(images_train[i])/255,labels_train[i]))
    
    with open("SAVE_files/flat_data_train.pkl", "wb") as f:
        pickle.dump(flat_data_train,f)

    #write test data
    flat_data_test  = []
    for i in range(size_test):
        flat_data_test.append((np.concatenate(images_test[i])/255,labels_test[i]))

    with open("SAVE_files/flat_data_test.pkl", "wb") as f:
        pickle.dump(flat_data_test,f)



# Reading tuples from file written by 'pkl_write()'
#       Output: (flat_data_train, flat_data_test) as tuples like 'pkl_write()'
def pkl_read():
    with open("SAVE_files/flat_data_train.pkl", "rb") as f:
        flat_data_train = pickle.load(f)
    with open("SAVE_files/flat_data_test.pkl", "rb") as f:
        flat_data_test = pickle.load(f)

    return flat_data_train, flat_data_test