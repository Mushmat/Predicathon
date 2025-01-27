import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as ply

# Paths
fake_images_path = r"E:/IIITB/Predicathon/project/data/train/fake_cifake_images"
real_images_path = r"E:/IIITB/Predicathon/project/data/train/real_cifake_images"
test_images_path = r"E:/IIITB/Predicathon/project/data/test/test"
train_json_path = r"E:/IIITB/Predicathon/project/data/real_cifake_preds.json"
