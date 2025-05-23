from paddleocr import PaddleOCR
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import difflib
import fitz  # PyMuPDF
import numpy as np
import cv2
from pyzbar.pyzbar import decode

import re
import os

# === 1. Load PDF and extract first page as image ===
pdf_path = 'C:/Users/mghir/Downloads/trt.pdf'
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
image = image.convert("RGB")  # back to RGB

# === 3. Define relative regions ===
REFERENCE_WIDTH = 836
REFERENCE_HEIGHT = 540

regions_rel = {
    "traite_num":        (650/836, 70/540, 800/836, 90/540),
    "amount_digits":     (623/836, 120/540, 821/836, 140/540),
    "date_due":          (23/836, 71/540, 385/836, 97/540),
    "date_created":      (144/836, 269/540, 271/836, 300/540),
    "order_number":      (540/836, 70/540, 720/836, 120/540),
    "rib":               (231/836, 110/540, 590/836, 141/540),
    "drawer_name":       (13/836, 169/540, 210/836, 203/540),
    "amount_words":      (100/836, 200/540, 350/836, 250/540),
    "company_name":      (222/836, 200/540, 588/836, 221/540),
    "payer_name_address":(362/836, 352/540, 548/836, 450/540),
    "bank":              (650/836, 320/540, 835/836, 430/540),
    "signature_stamp":   (650/836, 430/540, 835/836, 540/540),
    "signature":         (14/836, 74/540, 212/836, 150/540),
    "signature_tireur":  (560/836, 385/540, 815/836, 445/540),
}

# === 4. Convert to pixel regions with proportional margins ===
regions = {}
for key, (x0_rel, y0_rel, x1_rel, y1_rel) in regions_rel.items():
    mx = int(0.01 * img_w)
    my = int(0.01 * img_h)
    x0 = int(x0_rel * img_w) - mx
    y0 = int(y0_rel * img_h) - my
    x1 = int(x1_rel * img_w) + mx
    y1 = int(y1_rel * img_h) + my
    regions[key] = (
        max(0, x0), max(0, y0),
        min(img_w, x1), min(img_h, y1)
    )

# === 5. Initialize OCR ===
ocr = PaddleOCR(use_angle_cls=True, lang='fr')

# === 6. OCR per region ===
np_image = np.array(image)
extracted = {}
os.makedirs("debug_regions", exist_ok=True)

for field, (x0, y0, x1, y1) in regions.items():
    field_crop = np_image[y0:y1, x0:x1]
    result = ocr.ocr(field_crop, cls=True)

    text = ''
    if result and isinstance(result[0], list):
        text = ' '.join([line[1][0] for line in result[0] if line and line[1]])
    else:
        print(f"⚠️  No OCR result in region: {field}")

    extracted[field] = text
    Image.fromarray(field_crop).save(f"debug_regions/{field}.png")

# === 7. Text cleaning ===
def clean_text(text, field):
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x20-\x7E]+', '', text)


    if field == "amount_digits":
        text = re.sub(r'[^\d.,]', '', text)

    elif field in ["date_due", "date_created"]:
        # Match multiple common date formats
        date_patterns = [
            r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b',  # 12/05/2024 or 12-05-24
            r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b',  # 2024-05-12
            r'\b\d{1,2}\s+[A-Za-zéèêîû]+\s+\d{4}\b',  # 12 May 2024
            r'\b[A-Za-zéèêîû]+\s+\d{1,2},\s*\d{4}\b',  # May 12, 2024
        ]

        matched = None
        for pattern in date_patterns:
            found = re.search(pattern, text)
            if found:
                matched = found.group(0)
                break

        text = matched if matched else text  # keep full text if no date matched

    elif field in ["date_due", "date_created"]:
        match = re.search(r'\d{1,2}-\w{3}-\d{2,4}', text)
        text = match.group(0) if match else text

    elif field in ["traite_num", "order_number"]:
        text = re.sub(r'\D', '', text)

    elif field == "rib":
        text = re.sub(r'[^\d]', '', text)

    elif field == "amount_words":
        valid_words = {
            'zéro', 'un', 'deux', 'trois', 'quatre', 'cinq', 'six', 'sept', 'huit', 'neuf',
            'dix', 'onze', 'douze', 'treize', 'quatorze', 'quinze', 'seize',
            'vingt', 'trente', 'quarante', 'cinquante', 'soixante',
            'soixante-dix', 'soixante et onze', 'quatre-vingt', 'quatre-vingts',
            'quatre-vingt-dix', 'cent', 'cents', 'mille', 'million', 'millions',
            'milliard', 'milliards', 'et', 'dinars', 'dinar'}
        text = text.lower()
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[^a-zàâçéèêëîïôûùüÿñæœ\s-]', '', text)
        words = text.split()
        filtered = []
        for word in words:
            match = difflib.get_close_matches(word, valid_words, n=1, cutoff=0.8)
            if match:
                filtered.append(match[0])
        text = ' '.join(filtered)

    return text

# === 8. Annotate image with results ===
draw = ImageDraw.Draw(image)
try:
    font = ImageFont.truetype("arial.ttf", 14)
except:
    font = ImageFont.load_default()

for field, rect in regions.items():
    x0, y0, x1, y1 = rect
    draw.rectangle(rect, outline="red", width=2)
    raw_text = extracted.get(field, "").strip()
    cleaned = clean_text(raw_text, field)
    if cleaned:
        label = cleaned if len(cleaned) < 40 else cleaned[:37] + "..."
        draw.text((x0, y0 - 15), f"{field}: {label}", fill="blue", font=font)

# === 9. Barcode detection ===
cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
barcodes = decode(cv_image)
print("\n📦 Detected Barcodes:")
for barcode in barcodes:
    data = barcode.data.decode('utf-8')
    btype = barcode.type
    x, y, w, h = barcode.rect
    print(f"- {btype}: {data}")
    cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(cv_image, f"{btype}: {data}", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

# === 10. Save results ===
image.save('C:/Users/mghir/downloads/annotated_ocr.jpg')
cv2.imwrite('C:/Users/mghir/downloads/barcode_detected.jpg', cv_image)
print("\n✅ Annotated images saved.")

# === 11. Print final cleaned results ===
print("\n📝 Extracted Fields:")
for k, v in extracted.items():
    cleaned = clean_text(v, k)
    print(f"{k:20s}: {cleaned}")
