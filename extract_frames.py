import os
import cv2

# Paths to original (real) and manipulated (fake) video folders
original_videos_path = r"E:/IIITB/Predicathon/project/data/DFD_original sequences"  # Replace with your actual path
manipulated_videos_path = r"E:/IIITB/Predicathon/project/data/DFD_manipulated_sequences/DFD_manipulated_sequences"  # Replace with your actual path

# Paths to save extracted frames
output_real_frames_path = r"E:/IIITB/Predicathon/project/data/frames/real"
output_fake_frames_path = r"E:/IIITB/Predicathon/project/data/frames/fake"