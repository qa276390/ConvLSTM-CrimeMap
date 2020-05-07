import gzip
import math
import numpy as np
import os
from PIL import Image
import random
import torch
import torch.utils.data as data

def load_mnist(root, mode):
    # Load MNIST dataset for generating training data.
    """
    path = os.path.join(root, 'train-images-idx3-ubyte.gz')
    with gzip.open(path, 'rb') as f:
        mnist = np.frombuffer(f.read(), np.uint8, offset=16)
        mnist = mnist.reshape(-1, 28, 28)
    return mnist
    """
    if mode=='train':
        path = os.path.join(root, 'trainset.npy')
    else:
        path = os.path.join(root, 'validset.npy')

    return np.load(path)

def load_test_set(root):
    # Load the fixed dataset
    """
    filename = 'mnist_test_seq.npy'
    path = os.path.join(root, filename)
    dataset = np.load(path)
    dataset = dataset[..., np.newaxis]
    return dataset
    """
    path = os.path.join(root, 'testset.npy')
    return np.load(path)

class MovingMNIST(data.Dataset):
    def __init__(self, root, mode, is_train, n_frames_input, n_frames_output, num_objects,
                 transform=None):
        '''
        param num_objects: a list of number of possible objects.
        '''
        super(MovingMNIST, self).__init__()

        self.dataset = None
        if is_train:
            self.dataset = load_mnist(root, mode)
            self.length =  len(self.dataset) - (n_frames_input + n_frames_output)
        else:
            self.dataset = load_fixed_set(root, False)
            self.length =  len(self.dataset) - (n_frames_input + n_frames_output)


        self.is_train = is_train
        self.num_objects = num_objects
        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output
        self.n_frames_total = self.n_frames_input + self.n_frames_output
        self.transform = transform


        # For generating data
        self.image_size = 64 
        #self.digit_size_ = 28
        self.digit_size_ = 15
        self.step_length_ = 0.1


    def __getitem__(self, idx):
        length = self.n_frames_input + self.n_frames_output
        if self.is_train:
            images = self.dataset[idx:idx+length,:,:]
        else:
            images = self.dataset[idx:idx+self.n_frames_input,:,:]

        # if self.transform is not None:
        #     images = self.transform(images)

        r = 1
        w = int(self.image_size / r)
        images  = images[:, :, :, np.newaxis]
        #print(images.shape)
        images = images.reshape((length, w, r, w, r)).transpose(0, 2, 4, 1, 3).reshape((length, r * r, w, w))

        input = images[:self.n_frames_input]
        if self.is_train:
            output = images[self.n_frames_input:length]
        else:
            output = []

        frozen = input[-1]
        # add a wall to input data
        # pad = np.zeros_like(input[:, 0])
        # pad[:, 0] = 1
        # pad[:, pad.shape[1] - 1] = 1
        # pad[:, :, 0] = 1
        # pad[:, :, pad.shape[2] - 1] = 1
        #
        # input = np.concatenate((input, np.expand_dims(pad, 1)), 1)

        output = torch.from_numpy(output).contiguous().float()
        input = torch.from_numpy(input).contiguous().float()
        # print()
        # print(input.size())
        # print(output.size())

        out = [idx, output, input, frozen, np.zeros(1)]
        return out

    def __len__(self):
        return self.length
