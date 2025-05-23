from ultralytics import YOLO

# Load the model
model = YOLO('./runs/medfactor/stamp-detection-ydrxo/exp/weights/best.pt')

# Perform inference on an image or folder
results = model.predict('C:/Users/mghir/Pictures/Screenshots/CAP1.png')

# Display results
print(results[0].show())