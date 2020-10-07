# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 14:31:49 2020

@author: Aleksandra
"""

import os
import shutil
import random

from imutils import paths

#Getting 180 X-Ray images of normal (healthy) lungs from the dataset
imagePaths = list(paths.list_images("chest_xray/train/NORMAL"))

# randomly sample the image paths
random.seed(42)
random.shuffle(imagePaths)
imagePaths = imagePaths[:180]

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
	# extract the filename from the image path and then construct the
	# path to the copied image file in the destination folder
	filename = imagePath.split(os.path.sep)[-1]
	outputPath = os.path.sep.join(["dataset/normal", filename])

	# copy the image from source to destination folder
	shutil.copy2(imagePath, outputPath)