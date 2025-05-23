from paddleocr import PaddleOCR
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
from ultralytics import YOLO
import difflib
import fitz  # PyMuPDF
import numpy as np
import cv2
from pyzbar.pyzbar import decode
import re
from pathlib import Path

# === 1. Load PDF and extract first page as image ===
pdf_path = 'C:/Users/mghir/downloads/TR4.pdf'
doc = fitz.open(pdf_path)
page = doc[0]
pix = page.get_pixmap(dpi=300)
image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
img_w, img_h = image.size

# === 2. Enhance image ===
image = image.convert("L")  # grayscale
enhancer = ImageEnhance.Contrast(image)
image = enhancer.enhance(2.5)
enhancer = ImageEnhance.Sharpness(image)
image = enhancer.enhance(3.0)
image = image.convert("RGB")

# === 3. Define relative regions ===
REFERENCE_WIDTH = 836
REFERENCE_HEIGHT = 540
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

# === 4. Convert to pixel regions ===
MARGIN = 20
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

# === 5. OCR ===
ocr = PaddleOCR(use_angle_cls=True, lang='fr')
np_image = np.array(image)
ocr_result = ocr.ocr(np_image, cls=True)[0]

# === 6. Helper ===
def point_in_rect(pt, rect):
    x, y = pt
    x0, y0, x1, y1 = rect
    return x0 <= x <= x1 and y0 <= y <= y1

# === 7. Extract text ===
extracted = {field: "" for field in regions}
for line in ocr_result:
    box, (text, score) = line
    cx = sum(p[0] for p in box) / 4
    cy = sum(p[1] for p in box) / 4
    for field, rect in regions.items():
        if point_in_rect((cx, cy), rect):
            extracted[field] += text + " "

# === 8. Clean text ===
def clean_text(text, field):
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x20-\x7E]+', '', text)

    if field == "amount_digits":
        text = re.sub(r'[^\d.,]', '', text)
    elif field in ["place_created", "place_created1"]:
        text = re.sub(r'\b[aA]\b', '', text)
    elif field == "company_name":
        text = re.sub(r'\b(Oui|Non|y1gas|py1gas)\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'[^a-zA-Z0-9\s\-\&]', '', text)
    elif field in ["date_due", "date_created", "date_due1", "date_created1"]:
        patterns = [
            r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b',
            r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b',
            r'\b\d{1,2}\s+[A-Za-zéèêîû]+\s+\d{4}\b',
            r'\b[A-Za-zéèêîû]+\s+\d{1,2},\s*\d{4}\b',
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                text = match.group(0)
                break
    elif field == "traite_num":
        text = re.sub(r'\D', '', text)
    elif field == "bank":
        text = re.sub(r'\bDomiciliation\b', '', text, flags=re.IGNORECASE)
    elif field == "amount_words":
        valid_words = {
            'zéro','un','deux','trois','quatre','cinq','six','sept','huit','neuf',
            'dix','onze','douze','treize','quatorze','quinze','seize','vingt','trente',
            'quarante','cinquante','soixante','soixante-dix','quatre-vingt','cent','mille',
            'million','milliard','et','dinars','dinar'
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

# === 9. Post-process fields ===
def better_date(d1, d2):
    if not d1 and d2:
        return d2
    if d1 and not d2:
        return d1
    if not d1 and not d2:
        return ""
    score1 = len(re.findall(r'[^a-zA-Z0-9]', d1))
    score2 = len(re.findall(r'[^a-zA-Z0-9]', d2))
    return d1 if score1 <= score2 else d2

def looks_like_place(text):
    return bool(re.match(r'^[a-zA-Z\s\-\.]+$', text)) and len(text) > 3

cleaned = {k: clean_text(v, k) for k, v in extracted.items()}
final_fields = {}
final_fields["place_created"] = cleaned["place_created1"] if looks_like_place(cleaned["place_created1"]) else cleaned["place_created"]
final_fields["date_due"] = better_date(cleaned["date_due"], cleaned["date_due1"])
final_fields["date_created"] = better_date(cleaned["date_created"], cleaned["date_created1"])
for k in cleaned:
    if k not in ["place_created", "place_created1", "date_due", "date_due1", "date_created", "date_created1"]:
        final_fields[k] = cleaned[k]

# === 10. Annotate original image ===
draw = ImageDraw.Draw(image)
try:
    font = ImageFont.truetype("arial.ttf", 14)
except:
    font = ImageFont.load_default()

for field, rect in regions.items():
    if field in ["place_created1", "date_due1", "date_created1"]:
        continue
    x0, y0, x1, y1 = rect
    label = final_fields.get(field, "")
    if label:
        draw.rectangle(rect, outline="red", width=2)
        draw.text((x0, y0 - 15), f"{field}: {label[:40]}...", fill="blue", font=font)

image.save('C:/Users/mghir/downloads/annotated_ocr.jpg')

# === 11. Apply YOLO to signature + signature_tireur crops ===
signature_model = YOLO("./runs/medfactor/stamp-detection-ydrxo/exp/weights/best.pt")
crop_dir = Path("C:/Users/mghir/downloads/signature_crops")
pred_dir = Path("C:/Users/mghir/downloads/signature_preds")
crop_dir.mkdir(parents=True, exist_ok=True)
pred_dir.mkdir(parents=True, exist_ok=True)

for field in ["signature", "signature_tireur"]:
    if field not in regions:
        continue
    x0, y0, x1, y1 = regions[field]
    crop = image.crop((x0, y0, x1, y1))
    crop_path = crop_dir / f"{field}.jpg"
    crop.save(crop_path)

    crop_cv = cv2.cvtColor(np.array(crop), cv2.COLOR_RGB2BGR)
    results = signature_model.predict(crop_cv, conf=0.1)

    crop_annotated = crop.copy()
    draw = ImageDraw.Draw(crop_annotated)
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            draw.rectangle((x1, y1, x2, y2), outline="green", width=2)
            draw.text((x1, y1 - 15), f"YOLO: {box.conf[0]:.2f}", fill="green", font=font)

    annotated_path = pred_dir / f"{field}_pred.jpg"
    crop_annotated.save(annotated_path)
    print(f"✅ YOLO prediction for {field} saved to {annotated_path}")

# === 12. Barcode Detection ===
barcodes = decode(np_image)
print("\n📦 Detected Barcodes:")
for barcode in barcodes:
    data = barcode.data.decode('utf-8')
    btype = barcode.type
    x, y, w, h = barcode.rect
    print(f"- {btype}: {data}")
    cv2.rectangle(np_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imwrite('C:/Users/mghir/downloads/barcode_detected.jpg', np_image)

# === 13. Print Extracted Fields ===
print("\n📝 Extracted Fields:")
for k, v in final_fields.items():
    print(f"{k:20s}: {v}")
