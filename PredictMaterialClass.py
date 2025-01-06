import torch
import cv2
import json
import os
from torch import nn
import torch.nn.functional as F
from winsound import Beep

# Define the MaterialClassifier class
class MaterialClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MaterialClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Capture an image from the webcam
def capture_webcam_image(image_path="webcam_image.png"):
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        raise RuntimeError("Error: Webcam not found or cannot be opened.")

    ret, frame = cam.read()
    cam.release()
    cv2.destroyAllWindows()

    if not ret:
        raise RuntimeError("Error: Could not capture an image from the webcam.")

    cv2.imwrite(image_path, frame)
    print(f"Image successfully saved as {image_path}")
    return image_path


# Preprocess the image to extract RGB and HSV features from the first row
def preprocess_row(image_path):
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise ValueError("Error: Image not found or cannot be read.")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

    row_rgb = image_rgb[0, :640]
    row_hsv = image_hsv[0, :640]

    features = []
    for i in range(row_rgb.shape[0]):
        rgb = row_rgb[i]
        hsv = row_hsv[i]
        features.append(list(rgb) + list(hsv))

    return torch.tensor(features, dtype=torch.float32)


# Load the trained model and label mapping
def load_model_and_mapping(model_path="material_classifier.pth", label_mapping_path="label_mapping.json"):
    with open(label_mapping_path, "r") as f:
        label_mapping = json.load(f)
    reverse_mapping = {v: k for k, v in label_mapping.items()}

    input_size = 6
    num_classes = len(label_mapping)
    model = MaterialClassifier(input_size, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    return model, reverse_mapping


# Predict the material and aggregate probabilities for the entire image
def predict_material_with_certainty(image_path, model, reverse_mapping, threshold=0.5):
    features = preprocess_row(image_path)
    outputs = model(features)
    probabilities = F.softmax(outputs, dim=1)  # Convert outputs to probabilities

    # Aggregate probabilities over all pixels
    aggregated_probabilities = probabilities.mean(dim=0)  # Average over all pixels
    material_probabilities = {reverse_mapping[i]: float(prob) for i, prob in enumerate(aggregated_probabilities)}

    # Check for certainty
    most_likely_material = max(material_probabilities, key=material_probabilities.get)
    max_probability = material_probabilities[most_likely_material]

    if max_probability < threshold:  # If no material exceeds the threshold, return "Unsure"
        return "Unsure", material_probabilities

    return most_likely_material, material_probabilities


# Main function
if __name__ == "__main__":
    try:
        image_path = capture_webcam_image()
        model, reverse_mapping = load_model_and_mapping()
        result, probabilities = predict_material_with_certainty(image_path, model, reverse_mapping, threshold=0.5)

        if result == "Unsure":
            print("The model is unsure about the material.")
        else:
            print(f"The most likely material is: {result}")

        print("\nAggregated probabilities for the entire image:")
        for material, probability in probabilities.items():
            print(f"  {material}: {probability * 100:.2f}%")
    except Exception as e:
        print(str(e))

Beep(650, 250) # Notification
print("Done!")