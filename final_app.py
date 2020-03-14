# -*- coding: utf-8 -*-
"""
Face detection and recognition inference app

The following was adopted from the example illustrated on how to
use the `facenet_pytorch` python package to perform face detection
and recogition on an image dataset using an Inception Resnet V1 pretrained
on the VGGFace2 dataset.

This model is only tested where only 1 face per picture is found. For further
cleaning of pictures, use the cropping  pipeline to crop the face out of the picture
first
"""

from tkinter import filedialog
import tkinter as tk
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import pandas as pd
import os

# Checks if the system is NT liniage of Windows system. If so, workers = 0,
# workers denotes the number of processes that generate batches in parallel.
# A high enough number of workers assures that CPU computations are efficiently managed.
# see https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
workers = 0 if os.name == 'nt' else 4

"""#### Determine if an nvidia GPU is available"""
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

# Adding an MTCNN with default parameters
mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)

# Defining a InceptionResnetV1 model, pretrained on vggface2 dataset in evaluation mode on GPU
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

"""#### Define a dataset and data loader"""
# Defining a dataset and data loader: The input to collate_fn is a list of
# tensors with the size of batch_size, and the collate_fn function packs
# them into a mini-batch. In our case, the batch size is the size of the
# first item/image


def collate_fn(x):
    return x[0]


# Selecting the file path
root = tk.Tk()
root.withdraw()
print("Select the folder where the subfolders contain classes with images.")
file_path = filedialog.askdirectory(title="Select a path for classes")
# print("Selected folder path: " + file_path)

# Loading the images from the given path
dataset = datasets.ImageFolder(file_path)

# Assigns an index to the class name or a label for the ease of conversion later
dataset.idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}

# Loads the data in the dataset with the batch size defined by the collate function
# using the number of threads defined earlier
loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)

# Load the test data

# Selecting the file path
root = tk.Tk()
root.withdraw()
print("Select the folder where subfolders contain the images to be recognized.")
file_path = filedialog.askdirectory()
# print("Selected folder path: " + file_path)

testset = datasets.ImageFolder(file_path)
loader2 = DataLoader(testset, collate_fn=collate_fn, num_workers=workers)

"""#### Perfom MTCNN facial detection"""
# create empty lists
aligned = []
names = []
num_of_classes = 0

# x = PIL image and y  = index, x_aligned = image tensor, prob = prob that an image has a face
for x, y in loader:
    x_aligned, prob = mtcnn(x, return_prob=True)

    # if a face is detected
    if x_aligned is not None:
        num_of_classes = num_of_classes + 1
        # print('Face detected with probability: {:8f}'.format(prob))
        # the tensor is appended
        aligned.append(x_aligned)
        # the label of the tensor/image is appended
        names.append(dataset.idx_to_class[y])

# Save file names in a list

# Selecting the file path
root = tk.Tk()
root.withdraw()
print("Select the subfolder where the images to be recognized (.jpg or .jpeg) are located.")
file_path = filedialog.askdirectory()
# print("Selected folder path: " + file_path)
dirListing = os.listdir(file_path)

# Sorting the file, so the files are added to the list in order.
dirListing.sort()

imageNames = []

# add the file name to the imageName array
for item in dirListing:
    if ".jpg" or ".png" in item:
        # name = 'Person' + item[8:10]
        imageNames.append(item)
    else:
        print("A face was not detected in a class image! This image will not be processed further!")

# print('Actual image names:')
# print (imageNames)

# Passing the testing data into the model
count = 0
# classfor image to be recognized
for x, y in loader2:
    x_aligned, prob = mtcnn(x, return_prob=True)
    # print(x, y)
    # print(x_aligned)
    # Detects a face
    if x_aligned is not None:
        # print('Face detected with probability: {:8f}'.format(prob))
        aligned.append(x_aligned)
        names.append("Test" + imageNames[count])
        count = count + 1
    else:
        print("Face was not detected in a test image! This image will not be processed further!")
# print('Label names:')
# print(names)

"""#### Calculate image embeddings"""
# Stacking using the stack list the tensors to GPU - requires
# the dimensions to be the same for all stacked items
aligned = torch.stack(aligned).to(device)

# running the stacked data to resnet model and moves it to CPU from GPU if available
embeddings = resnet(aligned).detach().cpu()
# print(embeddings)

"""#### Print distance matrix for classes"""
# Calculating distance between each results
dists = [[(e1 - e2).norm().item() for e2 in embeddings] for e1 in embeddings]

# printing as pandas dataframe in a table
df = pd.DataFrame(dists, columns=names, index=names)
# print(df)

# Select the file path to save the distance result file
root = tk.Tk()
root.withdraw()
print("Select the folder where you would like to store the result csv file.")
file_path = filedialog.askdirectory()
print(file_path)
df.to_csv(file_path + '/result_dataframe.csv', index=True, header=True)
# print(pd.DataFrame(dists2, columns=names2, index=names2))

# find total number of rows in the matrix
length = 0
for i in names:
    length += 1

# Create an empty array of indexes for the images to be classified
min_idx_arr = [0] * (length - num_of_classes)  # tracks min distance index
identified_image_name = []  # tracks indentified image names
index = 0  # set index to 0

print('Printing matched results:')

# iterate through each row of distances 2D array
for dist in dists[num_of_classes:]:
    # print('dist is {}'.format(dist))

    # set initial values
    temp_min_dist = dist[0]
    image_name = 'unknown'
    # print(temp)

    # iterate through each column in each row for the classes to be recognized
    for j in range(num_of_classes):
        # print(dist[j])

        # find minimum values, index, and associated labels only within known class columns
        if dist[j] <= temp_min_dist:
            min_index = j
            min_idx_arr[index] = min_index
            temp_min_dist = dist[min_index]
            image_name = names[min_index]

    # If the minimum distance is 1 or more, then label the image as unknown
    if(dist[min_idx_arr[index]] >= 1):
        min_idx_arr[index] = -1
        image_name = 'unknown'

    # Append image name to the array
    identified_image_name.append(image_name)
    # print('min distance: ' + str(temp_min_dist) + 'at index: ' + str(min_idx_arr[index]))
    # print('image name before appending' + image_name)

    # Typical file names and classes are in the format "personne#####+angel+angel.jpg" format or "person##" format, so weeding out
    # cases where the files are intentionally added that would not match any class
    if not imageNames[index].startswith("person"):
        actual_image_name = imageNames[index]
    else:
        actual_image_name = 'Person' + imageNames[index][8:10]

    # print the file analysis on the terminal
    print('Model detected actual image of {} as '.format(
        actual_image_name) + identified_image_name[index])
    result = actual_image_name == names[min_idx_arr[index]]
    if result is True:
        print('The image is correctly recognized')
    else:
        print('The image was not recognized or an appropriate class did not exist!')

    # print(names[min_idx_arr[index]])
    index += 1
