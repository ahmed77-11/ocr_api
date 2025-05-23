import fitz  # PyMuPDF
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR
from transformers import LayoutLMForTokenClassification, LayoutLMTokenizerFast
import torch

# === 1. Load PDF and extract first page as image ===
pdf_path = 'C:/Users/mghir/downloads/TR4.pdf'
doc = fitz.open(pdf_path)
page = doc[0]
pix = page.get_pixmap(dpi=300)
image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
np_image = np.array(image)
image_width, image_height = image.size

# === 2. OCR with PaddleOCR ===
ocr = PaddleOCR(use_angle_cls=True, lang='fr')
ocr_result = ocr.ocr(np_image, cls=True)[0]

words, boxes = [], []
for line in ocr_result:
    box, (text, _) = line
    if not text.strip():
        continue
    words.append(text)
    x0 = min([p[0] for p in box])
    y0 = min([p[1] for p in box])
    x1 = max([p[0] for p in box])
    y1 = max([p[1] for p in box])
    boxes.append([int(x0), int(y0), int(x1), int(y1)])

# === 3. Normalize boxes to 0-1000 scale ===
norm_boxes = []
for box in boxes:
    x0, y0, x1, y1 = box
    norm_box = [
        int(1000 * x0 / image_width),
        int(1000 * y0 / image_height),
        int(1000 * x1 / image_width),
        int(1000 * y1 / image_height),
    ]
    norm_boxes.append(norm_box)

# === 4. Tokenizer and model setup ===
tokenizer = LayoutLMTokenizerFast.from_pretrained("microsoft/layoutlm-base-uncased")
model = LayoutLMForTokenClassification.from_pretrained("microsoft/layoutlm-base-uncased")

# === 5. Encode input ===
encoding = tokenizer(
    words,
    boxes=norm_boxes,
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=512
)

# === 6. Predict ===
with torch.no_grad():
    outputs = model(**encoding)
    predictions = outputs.logits.argmax(-1).squeeze().tolist()

tokens = tokenizer.convert_ids_to_tokens(encoding.input_ids.squeeze())

# === 7. Define label list (custom fields) ===
label_list = [
    "O",
    "B-TRAITE_NUM", "I-TRAITE_NUM",
    "B-AMOUNT_DIGITS", "I-AMOUNT_DIGITS",
    "B-DATE_DUE", "I-DATE_DUE",
    "B-DATE_CREATED", "I-DATE_CREATED",
    "B-ORDER_NUMBER", "I-ORDER_NUMBER",
    "B-RIB", "I-RIB",
    "B-DRAWER_NAME", "I-DRAWER_NAME",
    "B-AMOUNT_WORDS", "I-AMOUNT_WORDS",
    "B-COMPANY_NAME", "I-COMPANY_NAME",
    "B-PAYER_NAME_ADDRESS", "I-PAYER_NAME_ADDRESS",
    "B-BANK", "I-BANK",
    "B-SIGNATURE_STAMP", "I-SIGNATURE_STAMP",
    "B-SIGNATURE", "I-SIGNATURE",
    "B-SIGNATURE_TIREUR", "I-SIGNATURE_TIREUR"
]
id2label = {i: label for i, label in enumerate(label_list)}

# === 8. Print predictions with labels ===
print("\n📝 Tokens with predicted labels:")
for word, pred_id in zip(words, predictions[:len(words)]):
    label = id2label.get(pred_id, "O")
    print(f"{word:25s} => {label}")
