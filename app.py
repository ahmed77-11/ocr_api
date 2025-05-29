import shap
from flask import Flask, request, jsonify
from flask_cors import CORS

import os
import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
from google.cloud import vision
from google.api_core.exceptions import GoogleAPICallError
from num2words import num2words
from paddleocr import PaddleOCR
from ultralytics import YOLO
import numpy as np
import re
import difflib
import cv2
from pyzbar.pyzbar import decode
import tempfile
import datetime
import pandas as pd
import joblib
from sqlalchemy import create_engine
import base64
from io import BytesIO

app = Flask(__name__)
CORS(app,
     resources={r"/api/*": {
         "origins": "http://localhost:5173",
         "methods": ["POST", "GET", "PUT", "DELETE"],
         "allow_headers": ["Content-Type", "Authorization"]
}})

MODEL_PATH = './xgb_model56.pkl'
DB_URI = 'sqlite:///drafts1.db'
THRESHOLD = 0.667
# --- Load model ---
model = joblib.load(MODEL_PATH)

# --- Load per‑RIB & global stats once at startup ---
engine = create_engine(DB_URI)
df = pd.read_sql_table('drafts', engine)



rib_stats = (
    df
    .groupby('rib')['amount_digits']
    .agg(mean_amt='mean', std_amt='std', count='count')
)
pop_mean = df['amount_digits'].mean()
pop_std  = df['amount_digits'].std()

# Load the trained model
_xgb = joblib.load(MODEL_PATH)


def amount_to_words_fr(x): return num2words(x, lang='fr').replace('virgule','dinars zéro')

def is_valid_rib(v):
    s = ''.join(filter(str.isdigit, str(v)))
    if len(s) != 20: return False
    try:
        n = int(s[:-2] + '00'); chk = 97 - (n % 97)
        return chk == int(s[-2:])
    except: return False


def build_features(raw: dict) -> list:
    """Reproduce your notebook’s preprocessing pipeline."""
    # 1) Date gap
    d0 = pd.to_datetime(raw['date_created'])
    d1 = pd.to_datetime(raw['date_due'])
    gap = (d1 - d0).days

    # 2) Core flags & lengths
    amt = float(raw['amount_digits'])
    spell_ok = int(raw['amount_words'].strip().lower() == amount_to_words_fr(amt))
    sig_missing = int(not raw['signature_detected'])
    bc_bad = int(not raw['barcode_validates_traite'])
    print(is_valid_rib(raw['rib']))
    rib_bad = int(not is_valid_rib(raw['rib']))
    payer_len = len(raw['payer_name_address'])
    drawer_len = len(raw['drawer_name'])

    # 3) z‑score for this RIB
    r = raw['rib']
    if r in rib_stats.index and rib_stats.at[r, 'count'] >= 5:
        mu, sigma = rib_stats.at[r, 'mean_amt'], rib_stats.at[r, 'std_amt']
    else:
        mu, sigma = pop_mean, pop_std

    z = (amt - mu) / sigma if sigma and sigma > 0 else 0.0
    outlier_z = int(abs(z) > 3)

    return [amt, gap, spell_ok, sig_missing, bc_bad, rib_bad,
            payer_len, drawer_len, z, outlier_z]


explainer = shap.TreeExplainer(model)

# Feature names matching your model
features = ['amount_digits', 'gap_days', 'words_match', 'sig_missing', 'barcode_bad', 'rib_invalid', 'payer_len',
            'drawer_len', 'rib_amount_z', 'amount_incompatible']


def explain_prediction_api(raw_input, threshold=THRESHOLD):
    """
    Generate human-readable explanation for fraud prediction
    Returns structured explanation data for API response
    """
    try:
        # Build features using existing function
        feat = build_features(raw_input)

        # Get prediction probability
        prob = model.predict_proba([feat])[0][1]

        # Get SHAP values
        shap_vals = explainer.shap_values([feat])[0]

        # Generate explanations
        reasons_for_fraud = []
        reasons_against_fraud = []

        for name, val, sv in zip(features, feat, shap_vals):
            if sv > 0.01:  # Positive contribution to fraud
                if name == "sig_missing" and val == 1:
                    reasons_for_fraud.append({
                        "feature": "signature",
                        "message": "La signature n'est pas présente, ce qui indique souvent une fraude",
                        "impact": round(float(sv), 3)
                    })
                elif name == "words_match" and val == 0:
                    reasons_for_fraud.append({
                        "feature": "amount_match",
                        "message": "Le montant en lettres ne correspond pas au montant en chiffres",
                        "impact": round(float(sv), 3)
                    })
                elif name == "rib_invalid" and val == 1:
                    reasons_for_fraud.append({
                        "feature": "rib_validation",
                        "message": "Le format RIB semble invalide",
                        "impact": round(float(sv), 3)
                    })
                elif name == "barcode_bad" and val == 1:
                    reasons_for_fraud.append({
                        "feature": "barcode_validation",
                        "message": "Le code-barres n'est pas validé par rapport au numéro de traite",
                        "impact": round(float(sv), 3)
                    })
                elif name == "rib_amount_z" and abs(val) > 2:
                    reasons_for_fraud.append({
                        "feature": "amount_anomaly",
                        "message": f"Le montant s'écarte considérablement des normes historiques du RIB", #"(z-score: {val:.2f})"",
                        "impact": round(float(sv), 3)
                    })
                elif name == "amount_incompatible" and val == 1:
                    reasons_for_fraud.append({
                        "feature": "amount_compatibility",
                        "message": "RIB et montant sont statistiquement incompatibles",
                        "impact": round(float(sv), 3)
                    })
                elif name == "gap_days" and val > 365:
                    reasons_for_fraud.append({
                        "feature": "date_gap",
                        "message": f"Écart inhabituellement long entre la date de création et la date d'échéance ({int(val)} jours)",
                        "impact": round(float(sv), 3)
                    })
                elif name == "payer_len" and val < 10:
                    reasons_for_fraud.append({
                        "feature": "payer_name",
                        "message": "Le nom/l'adresse du tire semble incomplet ou trop court",
                        "impact": round(float(sv), 3)
                    })
                elif name == "drawer_len" and val < 10:
                    reasons_for_fraud.append({
                        "feature": "drawer_name",
                        "message": "Le nom du tireur semble incomplet ou trop court",
                        "impact": round(float(sv), 3)
                    })

            elif sv < -0.01:  # Negative contribution (against fraud)
                if name == "sig_missing" and val == 0:
                    reasons_against_fraud.append({
                        "feature": "signature",
                        "message": "La signature est présente, ce qui est courant dans les brouillons valides",
                        "impact": round(float(abs(sv)), 3)
                    })
                elif name == "words_match" and val == 1:
                    reasons_against_fraud.append({
                        "feature": "amount_match",
                        "message": "Les montants écrits et numériques correspondent parfaitement",
                        "impact": round(float(abs(sv)), 3)
                    })
                elif name == "rib_invalid" and val == 0:
                    reasons_against_fraud.append({
                        "feature": "rib_validation",
                        "message": "Le format RIB est valide",
                        "impact": round(float(abs(sv)), 3)
                    })
                elif name == "barcode_bad" and val == 0:
                    reasons_against_fraud.append({
                        "feature": "barcode_validation",
                        "message": "Code-barres validé correctement par rapport au numéro de trait",
                        "impact": round(float(abs(sv)), 3)
                    })
                elif name == "amount_digits" and 100 <= val <= 50000:
                    reasons_against_fraud.append({
                        "feature": "amount_range",
                        "message": "Le montant de la transaction se situe dans la fourchette normale",
                        "impact": round(float(abs(sv)), 3)
                    })
                elif name == "gap_days" and 0 <= val <= 180:
                    reasons_against_fraud.append({
                        "feature": "date_gap",
                        "message": f"L'écart entre les dates d'échéance est raisonnable ({int(val)} jours)",
                        "impact": round(float(abs(sv)), 3)
                    })

        # Generate summary
        if prob > threshold:
            summary = "Probablement frauduleux en raison de plusieurs signaux d'alarme détectés"
            risk_level = "HIGH"
        elif prob > threshold * 0.7:
            summary = "Risque de fraude modéré - nécessite un examen manuel"
            risk_level = "MEDIUM"
        else:
            summary = "Probablement légitime sur la base des indicateurs actuels"
            risk_level = "LOW"

        return {
            "probability": round(float(prob), 3),
            "predicted_label": bool(prob > threshold),
            "risk_level": risk_level,
            "summary": summary,
            "reasons_for_fraud": reasons_for_fraud,
            "reasons_against_fraud": reasons_against_fraud,
            "feature_values": {
                name: round(float(val), 3) if isinstance(val, (int, float)) else val
                for name, val in zip(features, feat)
            },
            "shap_values": {
                name: round(float(sv), 3)
                for name, sv in zip(features, shap_vals)
            }
        }

    except Exception as e:
        return {
            "error": f"Explanation generation failed: {str(e)}",
            "probability": None,
            "predicted_label": None
        }

# === Global Configuration ===
REFERENCE_WIDTH = 836
REFERENCE_HEIGHT = 540
MARGIN = 20

month_map = {
    'janvier': 1, 'février': 2, 'mars': 3, 'avril': 4, 'mai': 5, 'juin': 6,
    'juillet': 7, 'août': 8, 'septembre': 9, 'octobre': 10, 'novembre': 11, 'décembre': 12,
    'jan': 1, 'fév': 2, 'mar': 3, 'avr': 4, 'mai': 5, 'jun': 6, 'jul': 7,
    'aoû': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'déc': 12
}
# Define relative regions
regions_rel = {
    "traite_num": (650 / 836, 70 / 540, 800 / 836, 90 / 540),
    "amount_digits": (623 / 836, 120 / 540, 821 / 836, 155 / 540),
    "date_due": (270 / 836, 270 / 540, 402 / 836, 313 / 540),
    "date_created": (145 / 836, 270 / 540, 270 / 836, 310 / 540),
    "date_due1": (230 / 836, 70 / 540, 385 / 836, 105 / 540),
    "date_created1": (402 / 836, 80 / 540, 550 / 836, 95 / 540),
    "place_created": (11 / 836, 275 / 540, 146 / 836, 310 / 540),
    "place_created1": (402 / 836, 50 / 540, 540 / 836, 70 / 540),
    "rib": (231 / 836, 110 / 540, 600 / 836, 161 / 540),
    "drawer_name": (13 / 836, 169 / 540, 210 / 836, 203 / 540),
    "amount_words": (20 / 836, 222 / 540, 790 / 836, 260 / 540),
    # "company_name": (226 / 836, 200 / 540, 610 / 836, 225 / 540),
    "company_name": (220 / 836, 192 / 540, 620 / 836, 226 / 540),
    "payer_name_address": (362 / 836, 352 / 540, 548 / 836, 450 / 540),
    "bank": (555 / 836, 320 / 540, 820 / 836, 380 / 540),
    "signature_tireur": (560 / 836, 385 / 540, 815 / 836, 445 / 540),
}

# Load YOLO model (adjust path as needed)
signature_model = YOLO("./runs/medfactor/stamp-detection-ydrxo/exp/weights/best.pt")


# === Helper Functions ===
def setup_google_environment():
    """Setup Google Cloud Vision API credentials"""
    creds_path = "C:/Users/mghir/Desktop/vision-458206-e466edc966f8.json"
    if not os.path.exists(creds_path):
        raise FileNotFoundError(f"Credentials file not found at: {creds_path}")
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = creds_path


def process_pdf_to_image(pdf_file):
    """Convert PDF first page to enhanced image"""
    # Read file content into memory
    pdf_content = pdf_file.read()

    # Create temporary file and write content
    tmp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file_path = tmp_file.name
            tmp_file.write(pdf_content)
            tmp_file.flush()

        # Load PDF and extract first page
        doc = fitz.open(tmp_file_path)
        page = doc[0]
        pix = page.get_pixmap(dpi=300)
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        doc.close()

    finally:
        # Clean up temp file
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                os.unlink(tmp_file_path)
            except (OSError, PermissionError):
                # If we can't delete immediately, try again after a short delay
                import time
                time.sleep(0.1)
                try:
                    os.unlink(tmp_file_path)
                except (OSError, PermissionError):
                    pass  # File will be cleaned up by system eventually

    # Enhance image
    img_w, img_h = image.size
    image = image.convert("L")
    image = ImageEnhance.Contrast(image).enhance(2.5)
    image = ImageEnhance.Sharpness(image).enhance(2.5)
    image = image.convert("RGB")

    return image, img_w, img_h


def calculate_regions(img_w, img_h):
    """Calculate pixel regions from relative coordinates"""
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
    return regions


def point_in_rect(pt, rect):
    """Check if point is inside rectangle"""
    x, y = pt
    x0, y0, x1, y1 = rect
    return x0 <= x <= x1 and y0 <= y <= y1
def validate_date_order(date_str):
    """Validate logical date component order and values"""
    try:
        parts = re.split(r'[-/\.]', date_str)
        if len(parts) != 3:
            return False

        # Handle textual months
        if parts[1].lower() in month_map:
            day = int(parts[0])
            month = month_map[parts[1].lower()]
            year = int(parts[2])
            if year < 100:  # Handle 2-digit year
                year += 2000 if year < 25 else 1900
        else:
            # Try different format permutations
            for fmt in ['%d-%m-%Y', '%Y-%m-%d', '%d/%m/%Y', '%Y/%m/%d']:
                try:
                    parsed = datetime.strptime(date_str, fmt)
                    day, month, year = parsed.day, parsed.month, parsed.year
                    break
                except ValueError:
                    continue
            else:
                return False

        # Validate component ranges
        if not (1 <= day <= 31):
            return False
        if not (1 <= month <= 12):
            return False
        if year < 1900 or year > 2100:
            return False

        return True
    except Exception:
        return False


def clean_text(text, field):
    """Clean extracted text based on field type"""
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x20-\x7E]+', '', text)
    rem="Contre cette lettre de change Protestable paver lordre"

    if field == "amount_digits":
        text = re.sub(r'[^\d.,]', '', text)
    elif field in ["place_created", "place_created1"]:
        text = re.sub(r'\b[aA]\b', '', text)
    elif field == "company_name":
        text = re.sub(r'\b(Oui|Non)\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(y1gas|py1gas|Contre cette lettre de change Protestable payer lordre)\b', '', text, flags=re.IGNORECASE)

        text = re.sub(r'[^a-zA-Z0-9\s\-\&]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
    elif field in ["date_due", "date_created", "date_due1", "date_created1"]:
        text = re.sub(r'^\s*(Le|A)\s*', '', text, flags=re.IGNORECASE)
        # Updated patterns with flexible spacing around separators
        patterns = [
            r'\b\d{1,2}\s*[-/]\s*\d{1,2}\s*[-/]\s*\d{2,4}\b',
            r'\b\d{4}\s*[-/]\s*\d{1,2}\s*[-/]\s*\d{1,2}\b',
            r'\b\d{1,2}\s+[A-Za-zéèêîû]+\s+\d{4}\b',
            r'\b\d{1,2}\s*[-/]\s*[A-Za-z]{3,}\s*[-/]\s*\d{2,4}\b',
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
            'zéro', 'un', 'deux', 'trois', 'quatre', 'cinq', 'six', 'sept', 'huit', 'neuf',
            'dix', 'onze', 'douze', 'treize', 'quatorze', 'quinze', 'seize', 'vingt', 'trente',
            'quarante', 'cinquante', 'soixante', 'soixante-dix', 'quatre-vingt', 'quatre-vingts',
            'quatre-vingt-dix', 'cent', 'cents', 'mille', 'million', 'millions', 'milliard', 'milliards',
            'et', 'dinars', 'dinar'
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



def better_date(d1, d2):
    """Enhanced date selection with validation prioritization"""

    def date_quality(date):
        if not date:
            return (0, 0)  # (validity, completeness)

        valid = 1 if validate_date_order(date) else 0
        components = len(re.split(r'[-/\.]', date))
        year_length = len(re.search(r'\d{4}', date).group()) if re.search(r'\d{4}', date) else 0
        return (valid, components + year_length)

    q1 = date_quality(d1)
    q2 = date_quality(d2)

    if q1 > q2:
        return d1
    elif q2 > q1:
        return d2

    # Fallback to string length
    return d1 if len(d1) >= len(d2) else d2


def looks_like_place(text):
    """Check if text looks like a place name"""
    keywords = ['tunis', 'paris', 'lyon', 'marseille', 'cairo', 'algiers', 'casablanca', 'rabat', "sfax"]
    if any(k in text.lower() for k in keywords):
        return True
    return bool(re.match(r'^[a-zA-Z\s\-\.]+$', text)) and len(text) > 3


def post_process_fields(cleaned):
    """Post-process extracted fields"""
    final_fields = {}
    final_fields["place_created"] = cleaned["place_created1"] if looks_like_place(cleaned["place_created1"]) else \
    cleaned["place_created"]
    final_fields["date_due"] = better_date(cleaned["date_due"], cleaned["date_due1"])
    final_fields["date_created"] = better_date(cleaned["date_created"], cleaned["date_created1"])
    for k in cleaned:
        if k not in ["place_created", "place_created1", "date_due", "date_due1", "date_created", "date_created1"]:
            final_fields[k] = cleaned[k]
    return final_fields


def detect_signature_and_barcodes(image):
    """Detect signatures using YOLO and barcodes using pyzbar"""
    img_for_detection = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # YOLO signature detection
    results = signature_model(img_for_detection)


    signature_detected = False
    signature_box = None

    for r in results:
        for box in r.boxes:
            signature_detected = True
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            signature_box = (x1, y1, x2, y2)
            break

    # Barcode detection
    barcodes = decode(img_for_detection)
    barcode_data_list = [barcode.data.decode('utf-8') for barcode in barcodes]

    return signature_detected, signature_box, barcode_data_list


def validate_barcode_traite(barcode_list, traite_num):
    """Check if any barcode matches the traite number"""
    if not barcode_list or not traite_num:
        return False

    # Clean traite_num for comparison
    clean_traite = re.sub(r'\D', '', traite_num)

    for barcode in barcode_list:
        clean_barcode = re.sub(r'\D', '', barcode)
        if clean_barcode == clean_traite:
            return True
    return False


# === API Endpoint 1: Google Vision API ===
@app.route('/api/ocr/google', methods=['POST'])
def ocr_google_vision():
    """OCR using Google Vision API"""
    try:
        # Setup Google credentials
        setup_google_environment()

        # Check if file is uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Process PDF to image
        image, img_w, img_h = process_pdf_to_image(file)
        regions = calculate_regions(img_w, img_h)

        # Save image for OCR
        tmp_image_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_image:
                tmp_image_path = tmp_image.name
                image.save(tmp_image_path)

            # OCR with Google Vision API
            client = vision.ImageAnnotatorClient()
            with open(tmp_image_path, "rb") as image_file:
                content = image_file.read()
            vision_image = vision.Image(content=content)
            response = client.text_detection(vision_image)

            if response.error.message:
                raise GoogleAPICallError(response.error.message)

            ocr_texts = response.text_annotations

        finally:
            # Clean up temp file
            if tmp_image_path and os.path.exists(tmp_image_path):
                try:
                    os.unlink(tmp_image_path)
                except (OSError, PermissionError):
                    import time
                    time.sleep(0.1)
                    try:
                        os.unlink(tmp_image_path)
                    except (OSError, PermissionError):
                        pass

        # Extract text mapped to regions
        extracted = {field: "" for field in regions}
        for annotation in ocr_texts[1:]:  # skip full text at index 0
            vertices = annotation.bounding_poly.vertices
            cx = sum(v.x for v in vertices) / 4
            cy = sum(v.y for v in vertices) / 4
            for field, rect in regions.items():
                if point_in_rect((cx, cy), rect):
                    extracted[field] += annotation.description + " "

        # Clean extracted text
        cleaned = {k: clean_text(v, k) for k, v in extracted.items()}
        final_fields = post_process_fields(cleaned)

        # Detect signatures and barcodes
        signature_detected, signature_box, barcode_data_list = detect_signature_and_barcodes(image)
        final_fields["signature_detected"] = signature_detected
        final_fields["barcodes"] = barcode_data_list

        # Validate barcode against traite number
        barcode_matches_traite = validate_barcode_traite(barcode_data_list, final_fields.get("traite_num", ""))
        final_fields["barcode_matches_traite"] = barcode_matches_traite

        return jsonify({
            'success': True,
            'ocr_method': 'Google Vision API',
            'extracted_data': final_fields,
            'signature_detected': signature_detected,
            'signature_box': signature_box,
            'barcodes_detected': barcode_data_list,
            'barcode_validates_traite': barcode_matches_traite
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# === API Endpoint 2: PaddleOCR ===
@app.route('/api/ocr/paddle', methods=['POST'])
def ocr_paddle():
    """OCR using PaddleOCR"""
    try:
        # Check if file is uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Process PDF to image
        image, img_w, img_h = process_pdf_to_image(file)
        regions = calculate_regions(img_w, img_h)

        # OCR with PaddleOCR
        ocr = PaddleOCR(use_angle_cls=True, lang='fr')
        np_image = np.array(image)
        ocr_result = ocr.ocr(np_image, cls=True)[0]

        # Extract text mapped to regions
        extracted = {field: "" for field in regions}
        for line in ocr_result:
            box, (text, score) = line
            cx = sum(p[0] for p in box) / 4
            cy = sum(p[1] for p in box) / 4
            for field, rect in regions.items():
                if point_in_rect((cx, cy), rect):
                    extracted[field] += text + " "

        # Clean extracted text
        cleaned = {k: clean_text(v, k) for k, v in extracted.items()}
        final_fields = post_process_fields(cleaned)
        # for i in final_fields:
        #     print("done")
        #     print(final_fields[i])

        # Detect signatures and barcodes
        signature_detected, signature_box, barcode_data_list = detect_signature_and_barcodes(image)
        final_fields["signature_detected"] = signature_detected
        final_fields["barcodes"] = barcode_data_list

        # Validate barcode against traite number
        barcode_matches_traite = validate_barcode_traite(barcode_data_list, final_fields.get("traite_num", ""))
        final_fields["barcode_matches_traite"] = barcode_matches_traite

        return jsonify({
            'success': True,
            'ocr_method': 'PaddleOCR',
            'extracted_data': final_fields,
            'signature_detected': signature_detected,
            'signature_box': signature_box,
            'barcodes_detected': barcode_data_list,
            'barcode_validates_traite': barcode_matches_traite
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict/fraud', methods=['POST'])
def predict_fraud():
    """
    Enhanced fraud prediction with detailed explanations
    Expects JSON with:
      - amount_digits (str or float)
      - date_created  (YYYY-MM-DD)
      - date_due      (YYYY-MM-DD)
      - amount_words  (str)
      - signature_detected       (bool)
      - barcode_validates_traite (bool)
      - rib           (str)
      - payer_name_address (str)
      - drawer_name   (str)
    """
    data = request.get_json(force=True)

    # Check if explanation is requested (default: True)
    include_explanation = data.get('include_explanation', True)

    try:
        if include_explanation:
            # Get detailed explanation
            explanation = explain_prediction_api(data)
            print(explanation)
            return jsonify({
                'success': True,
                'prediction': {
                    'fraud_score': explanation['probability'],
                    'fraud_label': explanation['predicted_label'],
                    'risk_level': explanation['risk_level'],
                    'summary': explanation['summary'],
                    'detailed_analysis': {
                        'reasons_for_fraud': explanation['reasons_for_fraud'],
                        'reasons_against_fraud': explanation['reasons_against_fraud']
                    },
                    'technical_details': {
                        'feature_values': explanation['feature_values'],
                        'shap_contributions': explanation['shap_values']
                    }
                }
            })
        else:
            # Simple prediction without explanation (faster)
            feats = build_features(data)
            raw_model_prob = model.predict_proba([feats])[0][1]
            prob_python = float(raw_model_prob)
            score = round(prob_python, 3)
            label = bool(prob_python > THRESHOLD)

            return jsonify({
                'success': True,
                'prediction': {
                    'fraud_score': score,
                    'fraud_label': label
                }
            })

    except Exception as e:
        print(e)
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/explain/fraud', methods=['POST'])
def explain_fraud():
    """
    Dedicated endpoint for fraud explanation
    Same input format as prediction endpoint
    """
    data = request.get_json(force=True)

    try:
        explanation = explain_prediction_api(data)

        if 'error' in explanation:
            return jsonify({'success': False, 'error': explanation['error']}), 400

        return jsonify({
            'success': True,
            'explanation': explanation
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

# === Health Check Endpoint ===
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'available_endpoints': [
            '/api/ocr/google - POST - OCR using Google Vision API',
            '/api/ocr/paddle - POST - OCR using PaddleOCR',
            '/api/predict/fraud -POST- Predict Fraud',
            '/api/health - GET - Health check'
        ]
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)