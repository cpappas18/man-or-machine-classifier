#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import numpy as np
import imageio

# unpacking the data set 
def unpickle(file):
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data

humans = ['man', 'woman', 'boy', 'girl']
machines = ['telephone', 'television']

########################## processing training images ###########################

training_set = unpickle('./cifar-100-python/train')
label_names = unpickle('./cifar-100-python/meta')
fine_label_names = [t.decode('utf8') for t in label_names[b'fine_label_names']]
fine_labels = training_set[b'fine_labels']
file_names = [t.decode('utf8') for t in training_set[b'filenames']]
image_channels = training_set[b'data']

images = list()
for c in image_channels:
    image = np.zeros((32,32,3), dtype=np.uint8)
    image[...,0] = np.reshape(c[:1024], (32,32)) # red channel
    image[...,1] = np.reshape(c[1024:2048], (32,32)) # green channel
    image[...,2] = np.reshape(c[2048:], (32,32)) # blue channel
    images.append(image)

with open('./train_data.csv', 'w+') as f:
    for index,image in enumerate(images):
        filename = file_names[index]
        label = fine_labels[index]
        label = fine_label_names[label]        
        if (label in humans) or (label in machines): 
            if label in humans: 
                label = 'human'
            elif label in machines:
                label = 'machine'
            imageio.imwrite('./images/training/%s' %filename, image)
            f.write('images/training/%s,%s\n'%(filename,label))
     
########################## processing testing images ###########################

testing_set = unpickle('./cifar-100-python/test')
file_names = [t.decode('utf8') for t in testing_set[b'filenames']]
fine_labels = testing_set[b'fine_labels']
image_channels = testing_set[b'data']

images = list()
for c in image_channels:
    image = np.zeros((32,32,3), dtype=np.uint8)
    image[...,0] = np.reshape(c[:1024], (32,32)) # red channel
    image[...,1] = np.reshape(c[1024:2048], (32,32)) # green channel
    image[...,2] = np.reshape(c[2048:], (32,32)) # blue channel
    images.append(image)
    
with open('./test_data.csv', 'w+') as f:
    for index,image in enumerate(images):
        filename = file_names[index]
        label = fine_labels[index]
        label = fine_label_names[label]
        if (label in humans) or (label in machines): 
            if label in humans: 
                label = 'human'
            elif label in machines:
                label = 'machine'
            imageio.imwrite('./images/testing/%s' %filename, image)
            f.write('images/testing/%s,%s\n'%(filename,label))