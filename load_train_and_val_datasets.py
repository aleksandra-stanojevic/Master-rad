import cv2
import pandas as pd
import numpy as np
import random

from ultralytics import YOLO

origin_covid19 = "covid-chestxray-dataset/images/"
origin_pneumonia_and_no_findings = "CXR8/images/"
target_covid19 = "train_and_val_datasets/covid19/"
target_pneumonia = "train_and_val_datasets/pneumonia/"
target_no_findings = "train_and_val_datasets/no_findings/"

# Loading trained YOLO detection model
lung_model = YOLO("YOLO/lung_yolo_runs/lung_detector/weights/best.pt")

def crop_lungs_with_yolo(model, img):
    results = model(img)[0]

    if len(results.boxes) == 0:
        # No lungs detected → return original image
        return img

    boxes = results.boxes.xyxy.cpu().numpy()
    
    # Take the largest box (the lungs)
    areas = (boxes[:,2] - boxes[:,0]) * (boxes[:,3] - boxes[:,1])
    box = boxes[areas.argmax()]

    x1, y1, x2, y2 = map(int, box)
    cropped = img[y1:y2, x1:x2]

    return cropped

# Function to apply CLAHE to an image
def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(image)
    return clahe_image

# Reading the covid19 dataset metadata files
metadata = pd.read_csv("covid-chestxray-dataset/metadata.csv")

# Creating a DataFrame with X-ray PA views with covid19 finding
covid19_dataset = metadata.loc[(metadata['finding'] == 'Pneumonia/Viral/COVID-19') 
                              & (metadata['modality'] == 'X-ray')
                              & (metadata['view'] == 'PA')]
print(len(covid19_dataset))

# Copying all filtered images from origin folder to covid19 folder that will be used in further work
# Apply CLAHE to each image
for i, row in covid19_dataset.iterrows():
    covid19Image = cv2.imread(origin_covid19+row['filename'])
    # Crop lungs using YOLO 
    covid19Image = crop_lungs_with_yolo(lung_model, covid19Image)
    # CLAHE VERSION (CONVERT TO GRAYSCALE AND APPLY CLAHE)
    if len(covid19Image.shape) == 3:
        covid19Image = cv2.cvtColor(covid19Image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(target_covid19+row['filename'], apply_clahe(covid19Image))
    # cv2.imwrite(target_covid19+row['filename'], covid19Image)
print("Images for covid19 are loaded successfully")

# Reading the pneumonia and no_findings dataset metadata files
metadata = pd.read_csv("CXR8/metadata.csv")

# Creating a DataFrame with PA views with pneumonia
pneumonia_dataset = metadata.loc[(metadata['Finding Labels'].str.contains('Pneumonia')) 
                              & (metadata['View Position'] == 'PA')]
print(len(pneumonia_dataset))

# Copying all filtered images from origin folder to pneumonia folder that will be used in further work
# Apply CLAHE to each image
for i, row in pneumonia_dataset.iterrows():
    pneumoniaImage = cv2.imread(origin_pneumonia_and_no_findings+row['Image Index'])
    # Crop lungs using YOLO 
    pneumoniaImage = crop_lungs_with_yolo(lung_model, pneumoniaImage)
    # CLAHE VERSION (CONVERT TO GRAYSCALE AND APPLY CLAHE)
    if len(pneumoniaImage.shape) == 3:
        pneumoniaImage = cv2.cvtColor(pneumoniaImage, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(target_pneumonia+row['Image Index'], apply_clahe(pneumoniaImage))
    # cv2.imwrite(target_pneumonia+row['Image Index'], pneumoniaImage)
print("Images for pneumonia are loaded successfully")

# Creating a DataFrame with PA views with no_findings
no_findings_dataset = metadata.loc[(metadata['Finding Labels'] == 'No Finding') 
                              & (metadata['View Position'] == 'PA')]
print(len(no_findings_dataset))

# Getting all image paths for no_findings and randomly selecting 630 of them
imagePaths = list(no_findings_dataset['Image Index'])
random.seed(42)
random.shuffle(imagePaths)
imagePaths = imagePaths[:630]

# Copying all filtered images from origin folder to no_findings folder that will be used in further work
# Apply CLAHE to each image
for (i, imagePath) in enumerate(imagePaths):
    noFindingImage = cv2.imread(origin_pneumonia_and_no_findings+imagePath)
    # Crop lungs using YOLO 
    noFindingImage = crop_lungs_with_yolo(lung_model, noFindingImage)
    # CLAHE VERSION (CONVERT TO GRAYSCALE AND APPLY CLAHE)
    if len(noFindingImage.shape) == 3:
        noFindingImage = cv2.cvtColor(noFindingImage, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(target_no_findings+imagePath, apply_clahe(noFindingImage))
    # cv2.imwrite(target_no_findings+imagePath, noFindingImage)
print("Images for no_findings are loaded successfully")