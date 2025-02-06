import os
import cv2

# Paths to original (real) and manipulated (fake) video folders
original_videos_path = r"E:/IIITB/Predicathon/project/data/DFD/DFD_original sequences"  
manipulated_videos_path = r"E:/IIITB/Predicathon/project/data/DFD/DFD_manipulated_sequences/DFD_manipulated_sequences"  

# Paths to save extracted frames
output_real_frames_path = r"E:/IIITB/Predicathon/project/data/DFD/frames/real"
output_fake_frames_path = r"E:/IIITB/Predicathon/project/data/DFD/frames/fake"

# Ensure output directories exist
os.makedirs(output_real_frames_path, exist_ok=True)
os.makedirs(output_fake_frames_path, exist_ok=True)

def extract_frames(video_folder, output_folder, label, num_frames=50):
    """
    Extracts a fixed number of evenly spaced frames from each video.

    Args:
        video_folder (str): Path to the folder containing videos.
        output_folder (str): Path to save extracted frames.
        label (str): Label for the frames ('real' or 'fake').
        num_frames (int): Number of frames to extract per video.
    """
    video_files = os.listdir(video_folder)
    for video_file in video_files:
        video_path = os.path.join(video_folder, video_file)
        video_capture = cv2.VideoCapture(video_path)
        
        # Get total frame count
        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Select evenly spaced frames
        frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
        
        count = 0
        for frame_index in frame_indices:
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)  # Move to the frame
            success, frame = video_capture.read()
            if success:
                frame_name = f"{label}_{os.path.splitext(video_file)[0]}_frame_{count}.jpg"
                frame_path = os.path.join(output_folder, frame_name)
                cv2.imwrite(frame_path, frame)
                print(f"Saved: {frame_path}")
                count += 1

        video_capture.release()

# Extract more frames from real videos
extract_frames(original_videos_path, output_real_frames_path, label="real", num_frames=50)

# Extract fewer frames from fake videos
extract_frames(manipulated_videos_path, output_fake_frames_path, label="fake", num_frames=5)

print("Frame extraction completed!")
