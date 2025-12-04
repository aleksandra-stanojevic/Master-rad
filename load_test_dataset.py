import pandas as pd
import cv2
import os

from ultralytics import YOLO

origin_covid19 = "COVID_QU_Ex_Dataset/Lung Segmentation Data/Val/COVID-19/images/"
origin_pneumonia = "COVID_QU_Ex_Dataset/Lung Segmentation Data/Val/Non-COVID/images/"
origin_no_findings = "COVID_QU_Ex_Dataset/Lung Segmentation Data/Val/Normal/images/"
target_covid19 = "test_dataset/covid19/"
target_pneumonia = "test_dataset/pneumonia/"
target_no_findings = "test_dataset/no_findings/"

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


for filename in os.listdir(origin_covid19):
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):
        covid19Image = cv2.imread(os.path.join(origin_covid19, filename))
        # Crop lungs using YOLO 
        covid19Image = crop_lungs_with_yolo(lung_model, covid19Image)
        # CLAHE VERSION (CONVERT TO GRAYSCALE AND APPLY CLAHE)
        if len(covid19Image.shape) == 3:
            covid19Image = cv2.cvtColor(covid19Image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(target_covid19+row['filename'], apply_clahe(covid19Image))
        # cv2.imwrite(os.path.join(target_covid19, filename), covid19Image)
print("Images for covid19 are loaded successfully")

for filename in os.listdir(origin_pneumonia):
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):
        pneumoniaImage = cv2.imread(os.path.join(origin_pneumonia, filename))
        # Crop lungs using YOLO 
        pneumoniaImage = crop_lungs_with_yolo(lung_model, pneumoniaImage)
        # CLAHE VERSION (CONVERT TO GRAYSCALE AND APPLY CLAHE)
        if len(pneumoniaImage.shape) == 3:
            pneumoniaImage = cv2.cvtColor(pneumoniaImage, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(target_pneumonia+row['Image Index'], apply_clahe(pneumoniaImage))
        # cv2.imwrite(os.path.join(target_pneumonia, filename), pneumoniaImage)
print("Images for pneumonia are loaded successfully")

for filename in os.listdir(origin_no_findings):
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):
        noFindingImage = cv2.imread(os.path.join(origin_no_findings, filename))
        # Crop lungs using YOLO 
        noFindingImage = crop_lungs_with_yolo(lung_model, noFindingImage)
        # CLAHE VERSION (CONVERT TO GRAYSCALE AND APPLY CLAHE)
        if len(noFindingImage.shape) == 3:
            noFindingImage = cv2.cvtColor(noFindingImage, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(target_no_findings+imagePath, apply_clahe(noFindingImage))
        # cv2.imwrite(os.path.join(target_no_findings, filename), noFindingImage)
print("Images for no_findings are loaded successfully")