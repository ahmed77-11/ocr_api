from paddleocr import PaddleOCR
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import difflib
import fitz  # PyMuPDF
import numpy as np
import cv2
from pyzbar.pyzbar import decode
import re

# === 1. Load PDF and extract first page as image ===
pdf_path = 'C:/Users/mghir/downloads/TR3.pdf'
doc = fitz.open(pdf_path)
page = doc[0]
pix = page.get_pixmap(dpi=300)
image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
img_w, img_h = image.size

# === 2. Enhance image ===
image = image.convert("L")  # convert to grayscale (black and white)
enhancer = ImageEnhance.Contrast(image)
image = enhancer.enhance(2.5)
enhancer = ImageEnhance.Sharpness(image)
image = enhancer.enhance(3.0)
image = image.convert("RGB")  # convert back for drawing

# === 3. Define relative regions (based on 836x540 reference) ===
REFERENCE_WIDTH = 836
REFERENCE_HEIGHT = 540

regions_rel = {
    "traite_num":        (650/836, 70/540, 800/836, 90/540),
    "amount_digits":     (623/836, 120/540, 821/836, 155/540),
    "date_due":          (270/836, 270/540, 402/836, 313/540),
    "date_created":      (145/836, 270/540, 270/836, 310/540),
    "date_due1":      (230/836, 70/540, 385/836, 105/540),
    "date_created1":      (402/836, 80/540, 550/836, 95/540),
    "place_created":       (11/836,275/540,146/836,310/540),
    "place_created1":       (380/836,55/540,565/836,77/540),
    "order_number":      (540/836, 70/540, 720/836, 120/540),
    "rib":               (231/836, 110/540, 600/836, 161/540),
    "drawer_name":       (13/836, 169/540, 210/836, 203/540),
    "amount_words":      (20/836, 222/540, 790/836, 260/540),
    #"company_name":      (235/836, 200/540, 620/836, 225/540),
    "company_name":      (226/836, 200/540, 610/836, 225/540),
    "payer_name_address":(362/836, 352/540, 548/836, 450/540),
    "bank":              (650/836, 320/540, 835/836, 430/540),
    "signature_stamp":   (650/836, 430/540, 835/836, 540/540),
    "signature":         (14/836, 74/540, 212/836, 150/540),
    "signature_tireur":  (560/836, 385/540, 815/836, 445/540),
}

# === 4. Convert to pixel regions with margin ===
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

# === 5. Initialize OCR ===
ocr = PaddleOCR(use_angle_cls=True, lang='fr')

# === 6. OCR processing ===
np_image = np.array(image)
ocr_result = ocr.ocr(np_image, cls=True)[0]

# === 7. Helper: check point in region ===
def point_in_rect(pt, rect):
    x, y = pt
    x0, y0, x1, y1 = rect
    return x0 <= x <= x1 and y0 <= y <= y1

# === 8. Extract text per region ===
extracted = {field: "" for field in regions}
for line in ocr_result:
    box, (text, score) = line
    cx = sum(p[0] for p in box) / 4
    cy = sum(p[1] for p in box) / 4
    for field, rect in regions.items():
        if point_in_rect((cx, cy), rect):
            extracted[field] += text + " "

# === 9. Text cleaning ===
def clean_text(text, field):
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x20-\x7E]+', '', text)  # remove non-printable chars

    if field == "amount_digits":
        text = re.sub(r'[^\d.,]', '', text)

    elif field == "company_name":
        # Remove typical OCR noise like 'Oui', 'Non', and any non-alphanumeric garbage
        text = re.sub(r'\b(Oui|Non)\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(py1gas)\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'[^a-zA-Z0-9\s\-\&]', '', text)  # Remove special characters except dash and ampersand
        text = re.sub(r'\s+', ' ', text).strip()  # Normalize spacing



    elif field in ["date_due", "date_created"]:
        # Match multiple common date formats
        date_patterns = [
            r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b',  # 12/05/2024 or 12-05-24
            r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b',  # 2024-05-12
            r'\b\d{1,2}\s+[A-Za-zéèêîû]+\s+\d{4}\b',  # 12 May 2024
            r'\b[A-Za-zéèêîû]+\s+\d{1,2},\s*\d{4}\b',  # May 12, 2024
            r'\b\d{1,2}[-/][A-Za-z]{3,}[-/]\d{2,4}\b',  # 12-Oct-25 or 12-0ct-25 (bad OCR case)
        ]

        matched = None
        for pattern in date_patterns:
            found = re.search(pattern, text)
            if found:
                matched = found.group(0)
                break

        text = matched if matched else text  # keep full text if no date matched




    elif field == "traite_num":
        text = re.sub(r'\D', '', text)


    elif field == "amount_words":
        valid_words = {
            'zéro', 'un', 'deux', 'trois', 'quatre', 'cinq', 'six', 'sept', 'huit', 'neuf',
            'dix', 'onze', 'douze', 'treize', 'quatorze', 'quinze', 'seize',
            'vingt', 'trente', 'quarante', 'cinquante', 'soixante',
            'soixante-dix', 'soixante et onze', 'quatre-vingt', 'quatre-vingts',
            'quatre-vingt-dix', 'cent', 'cents', 'mille', 'million', 'millions',
            'milliard', 'milliards', 'et', 'dinars', 'dinar'}
        # Normalize text
        text = text.lower()
        text = re.sub(r'\d+', '', text)  # remove digits
        text = re.sub(r'[^a-zàâçéèêëîïôûùüÿñæœ\s-]', '', text)  # keep French letters only
        words = text.split()

        # Fuzzy match each word to closest valid French number word
        filtered = []
        for word in words:
            match = difflib.get_close_matches(word, valid_words, n=1, cutoff=0.8)
            if match:
                filtered.append(match[0])  # use corrected word

        text = ' '.join(filtered)
    elif field == "order_number":
        text = re.sub(r'\D', '', text)

    elif field == "rib":
        text = re.sub(r'[^\d]', '', text)

    # elif field in ["signature", "signature_stamp", "signature_tireur"]:
    #     text = ""  # ignore noisy signature zones

    # if len(text) > 60:
    #     text = text[:60] + "..."

    return text

# === 10. Annotate image ===
draw = ImageDraw.Draw(image)
try:
    font = ImageFont.truetype("arial.ttf", 14)
except:
    font = ImageFont.load_default()

for field, rect in regions.items():
    x0, y0, x1, y1 = rect
    draw.rectangle(rect, outline="red", width=2)
    raw_text = extracted[field].strip()
    cleaned = clean_text(raw_text, field)
    if cleaned:
        label = cleaned if len(cleaned) < 40 else cleaned[:37] + "..."
        draw.text((x0, y0 - 15), f"{field}: {label}", fill="blue", font=font)

# === 11. Detect barcodes ===
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

# === 12. Save annotated images ===
image.save('C:/Users/mghir/downloads/annotated_ocr.jpg')
cv2.imwrite('C:/Users/mghir/downloads/barcode_detected.jpg', cv_image)
print("\n✅ Annotated images saved.")

# === 13. Print cleaned extracted text ===
print("\n📝 Extracted Fields:")
for k, v in extracted.items():
    cleaned = clean_text(v, k)
    print(f"{k:20s}: {cleaned}")
