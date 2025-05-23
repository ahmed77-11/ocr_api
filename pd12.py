from paddleocr import PaddleOCR
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import fitz  # PyMuPDF
import numpy as np
import cv2
from pyzbar.pyzbar import decode

# === 1. Load PDF and extract first page as image ===
pdf_path = 'C:/Users/mghir/downloads/TR4.pdf'
doc = fitz.open(pdf_path)
page = doc[0]
pix = page.get_pixmap(dpi=300)
image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
img_w, img_h = image.size

# === 2. Define reference size and relative regions ===
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

# === 3. Initialize OCR ===
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# === 4. Helper: Convert relative region to pixel coordinates with scaling and margin ===
def get_region_coords(x0_rel, y0_rel, x1_rel, y1_rel, img_w, img_h, margin_x=15, margin_y=10):
    scale_x = img_w / REFERENCE_WIDTH
    scale_y = img_h / REFERENCE_HEIGHT
    x0 = int(x0_rel * REFERENCE_WIDTH * scale_x) - margin_x
    y0 = int(y0_rel * REFERENCE_HEIGHT * scale_y) - margin_y
    x1 = int(x1_rel * REFERENCE_WIDTH * scale_x) + margin_x
    y1 = int(y1_rel * REFERENCE_HEIGHT * scale_y) + margin_y
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(img_w, x1)
    y1 = min(img_h, y1)
    return (x0, y0, x1, y1)

# === 5. Adaptive margin OCR for one region ===
def ocr_region_adaptive(image, region, max_margin=30, step=5, min_text_length=3):
    """
    Try OCR on the region with increasing margin until enough text is found or max_margin reached.
    """
    x0_rel, y0_rel, x1_rel, y1_rel = region
    for margin in range(0, max_margin + 1, step):
        coords = get_region_coords(x0_rel, y0_rel, x1_rel, y1_rel, img_w, img_h, margin_x=margin, margin_y=margin)
        crop = image.crop(coords)
        # Enhance cropped region
        enhancer = ImageEnhance.Contrast(crop)
        crop_enhanced = enhancer.enhance(2.0)
        crop_enhanced = ImageEnhance.Sharpness(crop_enhanced).enhance(2.5)
        np_crop = np.array(crop_enhanced)
        result = ocr.ocr(np_crop, cls=True)

        if result and len(result) > 0 and result[0] is not None:
            text = " ".join(line[1][0] for line in result[0])
        else:
            text = ""

        if len(text.strip()) >= min_text_length:
            return text.strip()
    return ""  # If nothing found after max margin


# === 6. Extract text per region ===
extracted = {}
for field, rel_coords in regions_rel.items():
    extracted[field] = ocr_region_adaptive(image, rel_coords)

# === 7. Draw rectangles and labels on the original image ===
draw = ImageDraw.Draw(image)
try:
    font = ImageFont.truetype("arial.ttf", 14)
except:
    font = ImageFont.load_default()

for field, rel_coords in regions_rel.items():
    coords = get_region_coords(*rel_coords, img_w, img_h)
    draw.rectangle(coords, outline="red", width=2)
    text = extracted[field]
    if text:
        label = text if len(text) < 40 else text[:37] + "..."
        draw.text((coords[0], coords[1] - 15), f"{field}: {label}", fill="blue", font=font)

# === 8. Barcode detection using pyzbar on original image ===
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

# === 9. Save annotated images ===
image.save('C:/Users/mghir/downloads/annotated_ocr.jpg')
cv2.imwrite('C:/Users/mghir/downloads/barcode_detected.jpg', cv_image)
print("\n✅ Annotated images saved to downloads.")

# === 10. Print extracted fields ===
print("\n📝 Extracted Fields:")
for k, v in extracted.items():
    print(f"{k:20s}: {v}")
