from paddleocr import PaddleOCR
from PIL import Image, ImageDraw, ImageFont

# 1. Define relative regions based on a reference size
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

# 2. Initialize OCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# 3. Load image and compute actual pixel regions
img_path = 'C:/Users/mghir/downloads/TR4_page-0001.JPG'
image = Image.open(img_path).convert('RGB')
img_w, img_h = image.size

# Convert relative to absolute coordinates
regions = {
    key: (
        int(x0 * img_w),
        int(y0 * img_h),
        int(x1 * img_w),
        int(y1 * img_h)
    ) for key, (x0, y0, x1, y1) in regions_rel.items()
}

# 4. Run OCR
ocr_result = ocr.ocr(img_path, cls=True)[0]

# 5. Helper function: Check if point is inside a rectangle
def point_in_rect(pt, rect):
    x, y = pt
    x0, y0, x1, y1 = rect
    return x0 <= x <= x1 and y0 <= y <= y1

# 6. Extract text per region
extracted = {f: "" for f in regions}
for line in ocr_result:
    box, (text, score) = line
    cx = sum(p[0] for p in box) / 4
    cy = sum(p[1] for p in box) / 4
    for field, rect in regions.items():
        if point_in_rect((cx, cy), rect):
            extracted[field] += text + " "

# 7. Draw rectangles and text labels
draw = ImageDraw.Draw(image)
try:
    font = ImageFont.truetype("arial.ttf", 14)
except:
    font = ImageFont.load_default()

for field, rect in regions.items():
    x0, y0, x1, y1 = rect
    draw.rectangle(rect, outline="red", width=2)
    text = extracted[field].strip()
    if text:
        text_display = text if len(text) < 40 else text[:37] + "..."
        draw.text((x0, y0 - 15), f"{field}: {text_display}", fill="blue", font=font)

# 8. Save annotated image
output_path = 'C:/Users/mghir/downloads/annotated_test1.jpg'
image.save(output_path)
print(f"Annotated image with borders and text saved to: {output_path}")

# 9. (Optional) Print extracted values
print("\nExtracted fields:")
for k, v in extracted.items():
    print(f"{k:20s}: {v.strip()}")
