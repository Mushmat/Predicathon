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

#Load training Images
fake_images, fake_labels = load_images_from_folder(fake_images_path, label=0) #0 for fake
real_images, real_labels = load_images_from_folder(real_images_path, label=1)

#Combine fake and real data
train_images = np.array(fake_images + real_images)
train_labels = np.array(fake_labels + real_labels)

#Shuffle data to ensure randomness
shuffled_indices = np.random.permutation(len(train_images))
train_images = train_images[shuffled_indices]
train_labels = train_labels[shuffled_indices]