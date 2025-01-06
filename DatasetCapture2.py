import cv2
import numpy as np
import os
import json

# Capture an image from the webcam
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Error: Webcam not found or cannot be opened.")
    exit()

ret, frame = cam.read()
if ret:
    image_path: str = "webcam_image.png"
    cv2.imwrite(image_path, frame)
    print(f"Image successfully saved as {image_path}")
else:
    print("Error: Could not capture an image from the webcam.")

cam.release()
cv2.destroyAllWindows()

# Read the image and convert it to RGB format
image_bgr = cv2.imread(image_path)
if image_bgr is None:
    print("Error: Could not load the saved image.")
    exit()

# Generate RGB and HSV values for each pixel
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)  # Convert from BGR (normal output) to RGB
image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)  # Convert RGB to HSV

# Crop the image to one row (1-pixel height)
image_1px = image_rgb[0:1, :640]  # Height = 1, Width = 640
pixel_data = []


for col_idx in range(image_1px.shape[1]):  # Iterate over the width
    r, g, b = [f"{int(v):03}" for v in image_1px[0, col_idx]]  # RGB with leading zeros
    h, s, v = [f"{int(v):03}" for v in image_hsv[0, col_idx]]  # HSV with leading zeros
    pixel_data.append({
        "Pixel": f"{col_idx:03}",  # Pixel index formatted with leading zeros
        "RGB": {"R": r, "G": g, "B": b},
        "HSV": {"H": h, "S": s, "V": v}
    })

# Input the name of the material
material_name = input("Enter the name of the material: ")

# Create a dictionary to store the data in JSON format
data_entry = {material_name: {pixel["Pixel"]: {"RGB": pixel["RGB"], "HSV": pixel["HSV"]} for pixel in pixel_data}}

file_name = "Dataset.json"


def save_to_json(data, file_name):
    """
    Save the material data to a JSON file. If the material already exists, append without overwriting existing entries.
    """
    if not os.path.exists(file_name):
        # Create a new JSON file and write the initial data
        with open(file_name, "w") as file:
            json.dump([data], file, indent=4)  # Store data as a list of dictionaries
            print(f"File '{file_name}' created.")
    else:
        # Append new data to the existing JSON file
        with open(file_name, "r+") as file:
            try:
                existing_data = json.load(file)  # Load existing data
                if isinstance(existing_data, list):
                    existing_data.append(data)  # Append new material data
                else:
                    existing_data = [existing_data, data]  # Convert to a list if necessary
            except json.JSONDecodeError:
                existing_data = [data]  # Handle empty or corrupted files

            file.seek(0)  # Reset file pointer to the beginning
            json.dump(existing_data, file, indent=4)
            file.truncate()  # Remove any leftover content
            print(f"Data appended to '{file_name}'.")

save_to_json(data_entry, file_name)
print("Done!")
