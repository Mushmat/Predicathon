import os
import glob
import cv2
import albumentations as A

# --------------------------
# Directories
# --------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Gets the directory of the script
DATA_DIR = os.getenv("DATA_DIR", os.path.join(BASE_DIR, "data", "train"))

input_fake_dir = os.path.join(DATA_DIR, "fake_cifake_images")
input_real_dir = os.path.join(DATA_DIR, "real_cifake_images")

# Output directories (augmented dataset)
output_base_dir = os.getenv("OUTPUT_DIR", os.path.join(BASE_DIR, "data", "augmented"))
output_fake_dir = os.path.join(output_base_dir, "fake_cifake_images")
output_real_dir = os.path.join(output_base_dir, "real_cifake_images")

# Create output directories if they do not exist
os.makedirs(output_fake_dir, exist_ok=True)
os.makedirs(output_real_dir, exist_ok=True)

# --------------------------
# Augmentation Pipeline
# --------------------------
aug_pipeline = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=20, p=0.7),
    A.RandomResizedCrop(height=32, width=32, scale=(0.8, 1.0), ratio=(0.75, 1.33), p=0.7),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=0, p=0.7)
])

# Number of augmented images to generate per original image
num_augmented = 5

# --------------------------
# Augmentation Function
# --------------------------
def augment_and_save(input_dir, output_dir, class_label):
    image_files = glob.glob(os.path.join(input_dir, "*"))
    print(f"Found {len(image_files)} images in {input_dir} for class '{class_label}'.")
    
    for i, img_path in enumerate(image_files):
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (32, 32))
        
        base_filename = os.path.splitext(os.path.basename(img_path))[0]
        for j in range(num_augmented):
            augmented = aug_pipeline(image=img)
            aug_img = augmented["image"]
            aug_img_bgr = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
            output_filename = f"{base_filename}_aug{j+1}.jpg"
            output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_path, aug_img_bgr)
        
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1} images for class '{class_label}'.")

# --------------------------
# Augment Data for Each Class
# --------------------------
print("Starting augmentation for 'Fake' images...")
augment_and_save(input_fake_dir, output_fake_dir, "Fake")

print("Starting augmentation for 'Real' images...")
augment_and_save(input_real_dir, output_real_dir, "Real")

print("Data augmentation completed.")
