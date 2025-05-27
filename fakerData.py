import pandas as pd
import random
import numpy as np
from faker import Faker
from num2words import num2words

fake = Faker('fr_FR')
Faker.seed(42)
np.random.seed(42)


# Function to validate RIB using the JS logic translated to Python
def is_valid_rib(value):
    if not value:
        return False
    value = value.replace(" ", "").replace("-", "")
    if len(value) != 20:
        return False
    try:
        strN = value[:18] + '00'
        strCheck = value[-2:]
        big = int(strN)
        check = 97 - (big % 97)
        return int(strCheck) == check
    except Exception:
        return False


# Function to generate a valid RIB (brute-force method)
def generate_valid_rib():
    while True:
        base = ''.join([str(random.randint(0, 9)) for _ in range(18)])
        for i in range(100):  # try different check digits
            rib = base + f"{i:02}"
            if is_valid_rib(rib):
                return rib


# Generate French-formatted amount words from float
def amount_to_words(amount):
    return num2words(amount, lang='fr').replace("virgule", "dinars zéro")


def generate_fake_draft(fraud=False):
    amount = round(random.uniform(500, 10000), 3)
    amount_words = amount_to_words(amount)

    date_created_obj = fake.date_between(start_date="-1y", end_date="today")
    date_due_obj = fake.date_between(start_date=date_created_obj, end_date="+6m")
    date_created = date_created_obj.strftime('%d/%m/%Y')
    date_due = date_due_obj.strftime('%d/%m/%Y')

    if fraud:
        signature_detected = random.choice([False, False, True])
        barcode_validates = False
        company_name = fake.company() + " FAUX"
    else:
        signature_detected = True
        barcode_validates = True
        company_name = fake.company()

    return {
        "amount_digits": amount,
        "amount_words": amount_words,
        "bank": fake.company(),
        "company_name": company_name,
        "date_created": date_created,
        "date_due": date_due,
        "drawer_name": fake.name(),
        "payer_name_address": fake.address().replace("\n", " "),
        "place_created": fake.city(),
        "rib": generate_valid_rib(),
        "barcode_validates_traite": barcode_validates,
        "barcode_matches_traite": barcode_validates,
        "signature_detected": signature_detected,
        "traite_num": str(fake.random_number(digits=12)),
        "fraud_label": int(fraud)
    }


# Generate dataset
data = [generate_fake_draft(fraud=False) for _ in range(2500)] + \
       [generate_fake_draft(fraud=True) for _ in range(500)]

df = pd.DataFrame(data)
df.to_csv("synthetic_french_ribs.csv", index=False)
