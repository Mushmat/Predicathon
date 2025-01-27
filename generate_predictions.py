import json
from evaluate_model import get_predictions_and_names  # Import function from evaluate_model.py

# Get predictions and image names
predicted_labels, image_names = get_predictions_and_names()

# Map predictions to "real" and "fake"
predicted_classes = ["fake" if label == 0 else "real" for label in predicted_labels]

# Create JSON output
output_data = [{"index": img_name, "prediction": pred} for img_name, pred in zip(image_names, predicted_classes)]

# Save to a JSON file
output_json_path = "predictions.json"
with open(output_json_path, "w") as json_file:
    json.dump(output_data, json_file, indent=4)

print(f"Predictions saved to {output_json_path}")
