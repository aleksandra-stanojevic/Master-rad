# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 14:53:52 2020

@author: Aleksandra
"""

import pandas as pd
import os
import shutil

# importing the dataset and checking it's size
# and number of different (unique) findings

dataset = pd.read_csv("metadata.csv")
print(len(dataset))
print(len(dataset.finding.unique()))

# extracting just COVID-19 cases 
# in this dataset there are 45 different findings
# ['COVID-19' 'ARDS' 'SARS' 'Pneumocystis' 'Streptococcus' 'No Finding'
# 'Chlamydophila' 'E.Coli' 'COVID-19, ARDS' 'Klebsiella' 'Legionella' 
# 'Unknown' 'Pneumonia' 'Lipoid' 'Varicella' 'Bacterial'
# 'Mycoplasma Bacterial Pneumonia' 'Influenza'
# 'Cryptogenic Organizing Pneumonia' 'Lobar Pneumonia'
# 'Multilobar Pneumonia' 'Organizing Pneumonia' 'Eosinophilic Pneumonia'
# 'Unusual Interstitial Pneumonia' 'Lymphocytic Interstitial Pneumonia'
# 'Desquamative Interstitial Pneumonia' 'todo' 'Spinal Tuberculosis'
# 'Swine-Origin Influenza A (H1N1) Viral Pneumonia' 'Tuberculosis'
# 'Invasive Aspergillosis' 'Herpes pneumonia' 'Herpes pneumonia, ARDS'
# 'Accelerated Phase Usual Interstitial Pneumonia' 'Round pneumonia'
# 'Lymphocytic interstitial pneumonia'
# 'Allergic bronchopulmonary aspergillosis '
# 'Cryptogenic organising pneumonia' 'Chronic eosinophilic pneumonia'
# 'Aspiration pneumonia' 'Nocardia' 'MERS-CoV' 'Eosinophilic pneumonia'
# 'Cryptogenic organizing pneumonia' 'MRSA']

covid19_dataset = dataset.loc[dataset['finding'] == 'COVID-19']

# get just x-rays with Posterior-Anterior (PA) projection views
# in this dataset there are 6 different projection views
# anteroposterior (AP) views are not included
# CT axial projection views are not included
# CT coronal projection views are not included
# AP Supine views are not included
# L views are not included

covid19_dataset = covid19_dataset.loc[covid19_dataset['view'] == 'PA']
#%%
#Filtering only covid-19 X-Ray images from the dataset
for (i, row) in covid19_dataset.iterrows():
	# if (1) the current case is not COVID-19 or (2) this is not
	# a 'PA' view, then ignore the row
	if row["finding"] != "COVID-19" or row["view"] != "PA":
		continue

	# build the path to the input image file
	imagePath = os.path.sep.join(["covid-chestxray-dataset-master", "images",
		row["filename"]])

	# extract the filename from the image path and then construct the
	# path to the copied image file in a destination folder
	filename = row["filename"].split(os.path.sep)[-1]
	outputPath = os.path.sep.join(["dataset/covid19", filename])

	# copy the image from source to destination folder
	shutil.copy2(imagePath, outputPath)
   
# we now have 180 images in our destination folder
print(len(covid19_dataset))

	# if the input image file does not exist (there are some errors in
	# the COVID-19 metadeta file), ignore the row
	# if not os.path.exists(imagePath):
		# continue