import json
from evaluate_model import get_predictions_and_names

# Retrieve predictions and image names
predicted_labels, image_names = get_predictions_and_names()

# Map numeric labels to class names (adjust mapping as needed: here 0="fake", 1="real")
predicted_classes = ["fake" if label == 0 else "real" for label in predicted_labels]

# Create a list of dictionaries for each prediction
output_data = [{"index": img_name, "prediction": pred_class} 
               for img_name, pred_class in zip(image_names, predicted_classes)]

# Save predictions to a JSON file
output_json_path = "predictions.json"
with open(output_json_path, "w") as json_file:
    json.dump(output_data, json_file, indent=4)

print(f"Predictions saved to {output_json_path}")
