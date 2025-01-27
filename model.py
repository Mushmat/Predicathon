import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Paths
fake_images_path = r"E:/IIITB/Predicathon/project/data/train/fake_cifake_images"
real_images_path = r"E:/IIITB/Predicathon/project/data/train/real_cifake_images"
test_images_path = r"E:/IIITB/Predicathon/project/data/test/test"
train_json_path = r"E:/IIITB/Predicathon/project/data/real_cifake_preds.json"

# Loading images from folder and assign labels

def load_images_from_folder(folder_path, label):
    images = []
    labels = []
    for img_file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_file)
        if os.path.isfile(img_path):
            img = cv2.imread(img_path) #load image
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #convert to RGB
            images.append(img)
            labels.append(label) # 1 for real, 0 for fake
    return images, labels

