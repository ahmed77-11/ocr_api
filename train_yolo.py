from ultralytics import YOLO

# Load a base model (Nano for fast training)
model = YOLO('yolov8s.pt')  # use yolov8s.pt for more accuracy

# Train
model.train(
    data='./data/data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    project='runs/medfactor/stamp-detection-ydrxo',
    name='exp',
    workers=4
)

# Save best.pt will be in: runs/stamp_detect/exp/weights/best.pt

