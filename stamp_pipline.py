import os
import cv2
from ultralytics import YOLO
from paddleocr import PaddleOCR
from glob import glob

# Load model and OCR
model = YOLO("./runs/medfactor/stamp-detection-ydrxo/exp/weights/best.pt")  # path to your trained YOLOv8 model
ocr = PaddleOCR(use_angle_cls=True, lang='en')  # change lang if needed

def detect_and_read(image_path, output_dir="results/", conf_threshold=0.5):
    os.makedirs(output_dir, exist_ok=True)
    image = cv2.imread(image_path)
    results = model(image_path)[0]

    for i, (box, conf) in enumerate(zip(results.boxes.xyxy, results.boxes.conf)):
        if conf < conf_threshold:
            continue
        x1, y1, x2, y2 = map(int, box)
        cropped = image[y1:y2, x1:x2]

        # OCR on cropped image
        result = ocr.ocr(cropped, cls=True)
        text_lines = [line[1][0] for line in result[0]]
        full_text = "\n".join(text_lines)

        # Save cropped image and text
        base = os.path.splitext(os.path.basename(image_path))[0]
        stamp_name = f"{base}_stamp_{i+1}"
        cv2.imwrite(f"{output_dir}/{stamp_name}.png", cropped)
        with open(f"{output_dir}/{stamp_name}.txt", "w") as f:
            f.write(full_text)

        print(f"[✓] Saved {stamp_name}.png + OCR text")

def process_folder(input_dir="data/test/images", output_dir="results"):
    image_paths = glob(f"{input_dir}/*.jpg") + glob(f"{input_dir}/*.png")
    for path in image_paths:
        print(f"[→] Processing {path}")
        detect_and_read(path, output_dir)

if __name__ == "__main__":
    process_folder("data/test/images", "results")
