from ultralytics import YOLO

# Load a pre-trained YOLOv10n model
model = YOLO("best.pt")

# Perform object detection on an image
results = model("a.jpg")

# Display the results
results[0].show()