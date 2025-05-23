import os
import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
from google.cloud import vision
from google.api_core.exceptions import GoogleAPICallError
import numpy as np
import re
import difflib
import cv2
from pyzbar.pyzbar import decode
from ultralytics import YOLO

# === 1. Setup Google credentials ===
def setup_environment():
    creds_path = "C:/Users/mghir/Desktop/vision-458206-e466edc966f8.json"
    if not os.path.exists(creds_path):
        raise FileNotFoundError(f"Credentials file not found at: {creds_path}")
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = creds_path

# === 2. Load PDF, convert first page to image ===
pdf_path = 'C:/Users/mghir/downloads/TR1.pdf'
doc = fitz.open(pdf_path)
page = doc[0]
pix = page.get_pixmap(dpi=300)
image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
img_w, img_h = image.size

# === 3. Enhance image ===
image = image.convert("L")
image = ImageEnhance.Contrast(image).enhance(2.5)
image = ImageEnhance.Sharpness(image).enhance(3.0)
image = image.convert("RGB")

# === 4. Define relative regions ===
REFERENCE_WIDTH = 836
REFERENCE_HEIGHT = 540
MARGIN = 20
regions_rel = {
    "traite_num":        (650/836, 70/540, 800/836, 90/540),
    "amount_digits":     (623/836, 120/540, 821/836, 155/540),
    "date_due":          (270/836, 270/540, 402/836, 313/540),
    "date_created":      (145/836, 270/540, 270/836, 310/540),
    "date_due1":         (230/836, 70/540, 385/836, 105/540),
    "date_created1":     (402/836, 80/540, 550/836, 95/540),
    "place_created":     (11/836,275/540,146/836,310/540),
    "place_created1":    (402/836, 50/540, 540/836, 70/540),
    "rib":               (231/836, 110/540, 600/836, 161/540),
    "drawer_name":       (13/836, 169/540, 210/836, 203/540),
    "amount_words":      (20/836, 222/540, 790/836, 260/540),
    "company_name":      (226/836, 200/540, 610/836, 225/540),
    "payer_name_address":(362/836, 352/540, 548/836, 450/540),
    "bank":              (555/836, 320/540, 820/836, 380/540),
    "signature_tireur":  (560/836, 385/540, 815/836, 445/540),
}

regions = {}
for key, (x0_rel, y0_rel, x1_rel, y1_rel) in regions_rel.items():
    x0 = int(x0_rel * img_w) - MARGIN
    y0 = int(y0_rel * img_h) - MARGIN
    x1 = int(x1_rel * img_w) + MARGIN
    y1 = int(y1_rel * img_h) + MARGIN
    regions[key] = (
        max(0, x0), max(0, y0),
        min(img_w, x1), min(img_h, y1)
    )

# === 5. Save image for OCR ===
tmp_image_path = 'temp_ocr_image_google.jpg'
image.save(tmp_image_path)

# === 6. OCR with Google Vision API ===
def extract_text_blocks(img_path):
    client = vision.ImageAnnotatorClient()
    with open(img_path, "rb") as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    if response.error.message:
        raise GoogleAPICallError(response.error.message)
    return response.text_annotations

setup_environment()
ocr_texts = extract_text_blocks(tmp_image_path)

# === 7. Check if point is inside rect ===
def point_in_rect(pt, rect):
    x, y = pt
    x0, y0, x1, y1 = rect
    return x0 <= x <= x1 and y0 <= y <= y1

# === 8. Extract text mapped to regions ===
extracted = {field: "" for field in regions}
for annotation in ocr_texts[1:]:  # skip full text at index 0
    vertices = annotation.bounding_poly.vertices
    cx = sum(v.x for v in vertices) / 4
    cy = sum(v.y for v in vertices) / 4
    for field, rect in regions.items():
        if point_in_rect((cx, cy), rect):
            extracted[field] += annotation.description + " "

# === 9. Clean extracted text (same logic as your Paddle version) ===
def clean_text(text, field):
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x20-\x7E]+', '', text)

    if field == "amount_digits":
        text = re.sub(r'[^\d.,]', '', text)
    elif field in ["place_created", "place_created1"]:
        text = re.sub(r'\b[aA]\b', '', text)
    elif field == "company_name":
        text = re.sub(r'\b(Oui|Non)\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(y1gas|py1gas)\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'[^a-zA-Z0-9\s\-\&]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
    elif field in ["date_due", "date_created", "date_due1", "date_created1"]:
        patterns = [
            r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b',
            r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b',
            r'\b\d{1,2}\s+[A-Za-zéèêîû]+\s+\d{4}\b',
            r'\b[A-Za-zéèêîû]+\s+\d{1,2},\s*\d{4}\b',
            r'\b\d{1,2}[-/][A-Za-z]{3,}[-/]\d{2,4}\b',
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                text = match.group(0)
                break
    elif field == "traite_num":
        text = re.sub(r'\D', '', text)
    elif field == "bank":
        text = re.sub(r'\b(Domiciliation)\b', '', text, flags=re.IGNORECASE)
    elif field == "amount_words":
        valid_words = {
            'zéro','un','deux','trois','quatre','cinq','six','sept','huit','neuf',
            'dix','onze','douze','treize','quatorze','quinze','seize','vingt','trente',
            'quarante','cinquante','soixante','soixante-dix','quatre-vingt','quatre-vingts',
            'quatre-vingt-dix','cent','cents','mille','million','millions','milliard','milliards',
            'et','dinars','dinar'
        }
        text = text.lower()
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[^a-zàâçéèêëîïôûùüÿñæœ\s-]', '', text)
        words = text.split()
        filtered = [difflib.get_close_matches(w, valid_words, n=1, cutoff=0.8)[0]
                    for w in words if difflib.get_close_matches(w, valid_words, n=1, cutoff=0.8)]
        text = ' '.join(filtered)
    elif field == "rib":
        text = re.sub(r'[^\d]', '', text)

    return text

# === 10. Post-process some fields ===
def better_date(d1, d2):
    if not d1 and d2:
        return d2
    if d1 and not d2:
        return d1
    if not d1 and not d2:
        return ""
    score1 = len(re.findall(r'[^a-zA-Z0-9]', d1))
    score2 = len(re.findall(r'[^a-zA-Z0-9]', d2))
    if score1 < score2:
        return d1
    elif score2 < score1:
        return d2
    return d1 if len(d1) <= len(d2) else d2

def looks_like_place(text):
    keywords = ['tunis', 'paris', 'lyon', 'marseille', 'cairo', 'algiers', 'casablanca', 'rabat', "sfax"]
    if any(k in text.lower() for k in keywords):
        return True
    return bool(re.match(r'^[a-zA-Z\s\-\.]+$', text)) and len(text) > 3

cleaned = {k: clean_text(v, k) for k, v in extracted.items()}
final_fields = {}
final_fields["place_created"] = cleaned["place_created1"] if looks_like_place(cleaned["place_created1"]) else cleaned["place_created"]
final_fields["date_due"] = better_date(cleaned["date_due"], cleaned["date_due1"])
final_fields["date_created"] = better_date(cleaned["date_created"], cleaned["date_created1"])
for k in cleaned:
    if k not in ["place_created", "place_created1", "date_due", "date_due1", "date_created", "date_created1"]:
        final_fields[k] = cleaned[k]

# === 11. YOLO signature detection ===
signature_model = YOLO("./runs/medfactor/stamp-detection-ydrxo/exp/weights/best.pt")  # Adjust path accordingly
img_for_yolo = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
results = signature_model(img_for_yolo)[0]

print(results[0].show())

signature_detected = False
signature_box = None
for box in results.boxes:
    if box.cls == 0:  # assuming class 0 is signature
        signature_detected = True
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        signature_box = (x1, y1, x2, y2)
        break

final_fields["signature_detected"] = signature_detected

# === 12. Barcode detection (using pyzbar) ===
barcodes = decode(img_for_yolo)
barcode_data_list = [barcode.data.decode('utf-8') for barcode in barcodes]
final_fields["barcodes"] = barcode_data_list

# === 13. Annotate image ===
draw = ImageDraw.Draw(image)
font = ImageFont.load_default()

for field, txt in final_fields.items():
    if field in regions:
        x0, y0, x1, y1 = regions[field]
        draw.rectangle([x0, y0, x1, y1], outline="red", width=2)
        draw.text((x0, y0 - 15), f"{field}: {txt}", fill="blue", font=font)

if signature_detected and signature_box:
    draw.rectangle(signature_box, outline="green", width=3)
    draw.text((signature_box[0], signature_box[1] - 15), "Signature Detected", fill="green", font=font)

# === 14. Show image ===
image.show()

# === 15. Print final extracted values ===
print("Final extracted fields:")
for k, v in final_fields.items():
    if k != "barcodes":
        print(f"{k}: {v}")
print("Barcodes detected:", final_fields["barcodes"])
