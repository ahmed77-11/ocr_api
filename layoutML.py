# === INSTALLATION (run in terminal) ===
# pip install paddleocr paddlepaddle
# pip install transformers
# pip install pymupdf
# pip install torch
# pip install pillow
# pip install numpy

import fitz  # PyMuPDF
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR
from transformers import LayoutLMForTokenClassification, LayoutLMTokenizerFast
import torch

# === 1. Extract image from first page of PDF ===
pdf_path = 'C:/Users/mghir/downloads/TR4.pdf'
doc = fitz.open(pdf_path)
page = doc[0]
pix = page.get_pixmap(dpi=300)
image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
np_image = np.array(image)
image_width, image_height = image.size

# === 2. Run PaddleOCR ===
ocr = PaddleOCR(use_angle_cls=True, lang='fr')
ocr_result = ocr.ocr(np_image, cls=True)[0]

words = []
boxes = []
for line in ocr_result:
    box, (text, _) = line
    words.append(text)
    x0 = min([p[0] for p in box])
    y0 = min([p[1] for p in box])
    x1 = max([p[0] for p in box])
    y1 = max([p[1] for p in box])
    boxes.append([int(x0), int(y0), int(x1), int(y1)])

# === 3. Normalize bounding boxes for LayoutLM (0-1000 scale) ===
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

# === 4. Prepare tokenizer and model ===
tokenizer = LayoutLMTokenizerFast.from_pretrained("microsoft/layoutlm-base-uncased")
model = LayoutLMForTokenClassification.from_pretrained("microsoft/layoutlm-base-uncased")

# === 5. Tokenize words, align boxes to tokens ===
encoding = tokenizer(
    words,
    is_split_into_words=True,
    padding="max_length",
    truncation=True,
    max_length=512,
    return_offsets_mapping=True,
)

word_ids = encoding.word_ids()

token_boxes = []
for word_idx in word_ids:
    if word_idx is None:
        # Special tokens get a dummy bbox
        token_boxes.append([0, 0, 0, 0])
    else:
        token_boxes.append(norm_boxes[word_idx])

encoding["bbox"] = token_boxes

# Convert everything to tensors
input_ids = torch.tensor([encoding["input_ids"]])
attention_mask = torch.tensor([encoding["attention_mask"]])
bbox = torch.tensor([encoding["bbox"]])

# === 6. Predict with LayoutLM ===
with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, bbox=bbox)
    predictions = outputs.logits.argmax(-1).squeeze().tolist()

tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])

# === 7. Print tokens with predicted label IDs ===
print("\n📝 Tokens with predicted labels:")
for token, label in zip(tokens, predictions):
    print(f"{token:20s} => Label ID: {label}")

# === NOTE ===
# To get meaningful labels (e.g., B-AMOUNT, I-DATE), you need to fine-tune LayoutLM on your dataset,
# or map label IDs if you have a predefined label map.
