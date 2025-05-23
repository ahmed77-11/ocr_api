import os
from PIL import Image, ImageDraw, ImageFont
import pandas as pd

# Example field data
example = {
    "traite_num": "010755503657",
    "amount_digits": "10000.000",
    "date_due": "12-0ct-25",
    "date_created": "24-avrl25",
    "order_number": "20107555036573",
    "rib": "24000044729251230172",
    "drawer_name": "",
    "amount_words": "dix mille dinars",
    "company_name": "Non Oui MED FACTOR",
    "payer_name_address": "3STAR ELECTRONICS",
    "bank": "SOCIETE ATED TACTOR GP 1 & 12-Boumhell Tél: 70.020.500 Fax: 71.452.534 IC01",
    "signature_stamp": "u cJj X &X",
    "signature": "Signature du tir 000 3J8 043",
    "signature_tireur": "Tél: 70.020.500 Fax: 71.452.534 IC01 Signature u tir u cJj X &X"
}

# Directories
base_dir = os.path.abspath(os.path.dirname(__file__))
train_img_dir = os.path.join(base_dir, "train_images")
os.makedirs(train_img_dir, exist_ok=True)

labels = []
# load a default font (falls back to PIL’s built‑in if Arial isn’t installed)
try:
    font = ImageFont.truetype("arial.ttf", 24)
except IOError:
    font = ImageFont.load_default()

for field, text in example.items():
    img = Image.new("RGB", (800, 100), "white")
    draw = ImageDraw.Draw(img)
    draw.text((10, 35), text, font=font, fill="black")

    fname = f"{field}.jpg"
    path = os.path.join(train_img_dir, fname)
    img.save(path)

    # relative path + label
    labels.append({
        "image_path": f"train_images/{fname}",
        "text": text
    })

# Save to TSV (no header, tab‑separated)
labels_df = pd.DataFrame(labels)
labels_file = os.path.join(base_dir, "train_labels.tsv")
labels_df.to_csv(labels_file, sep="\t", index=False, header=False)

print(f"✅ Generated {len(labels)} images in {train_img_dir!r}")
print(f"✅ Labels file written to {labels_file!r}")
