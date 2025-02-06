
![Logo](https://i.ibb.co/tphQf7Cf/Picture1.png)


# 🛡️ Deepfake Detection using AI 🛡️

(**Final Submission - KFold Model**)

Welcome to our Deepfake Detection Model submission for the competition. Our AI-powered deepfake detection system analyzes images to distinguish between real and fake images with high accuracy. The model is trained using a K-Fold Cross-Validation approach, ensuring robustness and generalizability.

## 🎯 Objectives

- **Detect Deepfake Images:**: Accurately classify images as real or fake using advanced deep learning techniques.
- **Improve Model Robustness:**: Implement K-Fold Cross-Validation for better generalization.
- **Optimize for Performance:**: Use CNN and data augmentation for improved classification accuracy.
- **Provide an Easy-to-Use Solution:**: Ensure smooth deployment with a structured codebase.
  
## 🔍 Features

- **✅ High Accuracy:**: Trained with CNN for superior performance.

- **✅ K-Fold Cross-Validation:**: Enhances generalizability by training on multiple data splits.
- **✅ Data Augmentation:**: Ensures robustness against varied deepfake patterns.
- **✅ Scalability:**:  Designed to handle large datasets with batch processing.
- **✅ Optimized Performance:**: Uses early stopping and learning rate scheduling for efficient training.
- **✅ Structured Codebase:**: Includes a requirements.txt file and a clear setup guide.

## 📁 Folder Structure

`` Deepfake-Detection/ ``

``│── data/ # Data folder (Test images should be placed here)``
``│── models/  # Saved trained models``
``│── scripts/ # Python scripts for preprocessing & evaluation``
``├── train_model.py  # Training script``
``├── evaluate_model.py  # Evaluation script (to generate predictions)``
``│── outputs/  # Folder for storing generated JSON predictions``
``│── requirements.txt  # Dependencies for running the model``
``│── Spades_prediction.json # Final JSON file for submission``
``│── README.md # This file``
``│── Spades_presentation.pdf # Final Report + Presentation (single PDF) ``

## 🛠️ Running Requirements

Ensure you have Python 3.9+ installed on your system.

### Install Dependencies:
`` pip install -r requirements.txt ``

### Hardware Requirements:
- CPU: Minimum Intel i5 (Recommended: Intel i7 or AMD Ryzen 7)
- GPU (Optional, Recommended for training): NVIDIA RTX 3060+ (CUDA enabled)
- RAM: Minimum 8GB (Recommended: 16GB+)

## 📥 Installation & Running the Model

### Step 1: Clone the Repository
`` git clone https://github.com/Mushmat/Predicathon/tree/main ``
`` cd deepfake-detection ``

### Step 2: Install Dependencies
`` pip install -r requirements.txt ``

### Step 3: Place Test Images
Ensure all test images are stored in the /data/test/ folder.

### Step 4: Run the Model
Execute the evaluation script to generate predictions:

`` python scripts/evaluate_model.py ``

### Step 5: Check Predictions
The predictions will be saved in the outputs/teamname_prediction.json file.

## 📚 Model Details

### Model Architecture
- Backbone Model: CNN
- Input Size: 32x32
- Loss Function: Categorical Cross-Entropy
- Optimizer: Adam (Adaptive Learning Rate)
- Learning Rate Schedule: Cosine Decay with Warmup
- Regularization: L2 weight decay, dropout
- Data Augmentation: Albumentations library (random rotations, flips, brightness adjustments)

## 📊 Submission Components
Our final submission includes:
✅ 1. Predicted JSON File (Spades_prediction.json)
- Format:
- 
- `` [
    { "index": "1.png", "prediction": "fake" },
    { "index": "2.png", "prediction": "real" }
] ``

- Stored in ``/outputs/``

- ✅ 2. Final Report + Presentation (Spades_presentation.pdf)
  
- Includes methodology, preprocessing steps, model details, challenges, and results.
  
- ✅ 3. Code Repository (Optional)
  
- GitHub repository for reproducibility.
  

## 🚧 Challenges Faced

- **Data Imbalance**: Adjusted by class weighting and augmentation.
- **Overfitting**: Reduced using dropout layers, L2 regularization, and early stopping.
- **Computation Bottlenecks**: Optimized by batch normalization and mixed precision training.


## 🔮 Future Scope

-🔹 Improve Deepfake Generalization: Train on larger datasets for better generalizability.

-🔹 Enhance Detection with Video Input: Extend the model for real-time deepfake detection.

-🔹 Optimize Model for Mobile Deployment: Convert to TensorFlow Lite for mobile applications.

-🔹 Explainable AI: Integrate Grad-CAM to visualize model decision-making.


## ❓ FAQs

- 1. How do I run the model on my system?

- Follow the Installation & Running the Model section.

- 2. Can I train the model on my own dataset?

- Yes! Modify train_model.py and provide your dataset in /data/train/.

- 3. What format should the test images be in?

- Images should be JPEG/PNG, and stored in /data/test/.
   
## 🙌 Acknowledgements

-  🔹 TensorFlow: Framework used for model development.

- 🔹 Albumentations: Data augmentation techniques.

- 🔹 OpenAI & Research Papers: Reference materials.

- 🔹 GitHub & Community Support: Collaboration and resources.
  

🔗 Contact Us
For queries, reach out via GitHub Issues.

🚀 Spades | Deepfake Detection AI | Competition Submission 🚀

