import os
import cv2

# Paths to original (real) and manipulated (fake) video folders
original_videos_path = r"E:/IIITB/Predicathon/project/data/DFD_original sequences"  
manipulated_videos_path = r"E:/IIITB/Predicathon/project/data/DFD_manipulated_sequences/DFD_manipulated_sequences"  

# Paths to save extracted frames
output_real_frames_path = r"E:/IIITB/Predicathon/project/data/frames/real"
output_fake_frames_path = r"E:/IIITB/Predicathon/project/data/frames/fake"

def extract_frames(video_folder, output_folder, label, frame_rate=1):
    """
    Extracts frames from videos and saves them into the specified output folder.

    Args:
        video_folder (str): Path to the folder containing videos.
        output_folder (str): Path to save extracted frames.
        label (str): Label for the frames ('real' or 'fake').
        frame_rate (int): Extract every 'frame_rate'-th frame (1 = 1 frame per second).
    """
    video_files = os.listdir(video_folder)
    for video_file in video_files:
        video_path = os.path.join(video_folder, video_file)
        video_capture = cv2.VideoCapture(video_path)
        count = 0
        success, frame = video_capture.read()
        while success:
            if int(video_capture.get(cv2.CAP_PROP_POS_FRAMES)) % frame_rate == 0:
                # Save the frame with a descriptive name
                frame_name = f"{label}_{os.path.splitext(video_file)[0]}_frame_{count}.jpg"
                frame_path = os.path.join(output_folder, frame_name)
                cv2.imwrite(frame_path, frame)
                print(f"Saved: {frame_path}")
            success, frame = video_capture.read()
            count += 1
        video_capture.release()

# Extract frames from real videos
extract_frames(original_videos_path, output_real_frames_path, label="real", frame_rate=30)  # Adjust frame_rate as needed

# Extract frames from fake videos
extract_frames(manipulated_videos_path, output_fake_frames_path, label="fake", frame_rate=30)  # Adjust frame_rate as needed

print("Frame extraction completed!")