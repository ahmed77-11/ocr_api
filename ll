# fraud_pipeline_tf.py
"""
End-to-End Fraud Detection Pipeline for Bank Drafts using TensorFlow

Steps:
1. Synthetic Data Generation (prototyping)
2. Persist Historical Drafts into a Database (SQLite/Postgres)
3. Data Cleaning & Preprocessing
4. Build & Refresh Statistics (RIB-level, Bank-level, Population-level)
5. Train TensorFlow Models:
     - Autoencoder for anomaly detection on amounts
     - MLP classifier for supervised fraud detection
6. Serialize Models & Stats
7. Inference Function to score new OCR-extracted drafts
8. Onboard new RIB statistics after approval
9. Data Visualization

Usage:
  python fraud_pipeline_tf.py generate
  python fraud_pipeline_tf.py train --data <data_csv> --model-dir <model_dir>
  python fraud_pipeline_tf.py predict --ocr <ocr_json> --model-dir <model_dir>
  python fraud_pipeline_tf.py onboard --rib <rib> --amt <amount>
  python fraud_pipeline_tf.py visualize

Dependencies:
  pandas, numpy, faker, num2words, sqlalchemy, joblib, tensorflow
"""
import os
import json
import argparse
import random
import pandas as pd
import numpy as np
from faker import Faker
from num2words import num2words
from sqlalchemy import create_engine
import joblib
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# ---------- 0. Configuration ----------
DB_URI = 'sqlite:///drafts.db'  # or your PostgreSQL URI
ENGINE = create_engine(DB_URI)

# ---------- Utility Functions ----------
def is_valid_rib(v: str) -> bool:
    s = v.replace(' ', '').replace('-', '')
    if len(s) != 20:
        return False
    n = int(s[:-2] + '00')
    chk = 97 - (n % 97)
    return chk == int(s[-2:])


def amount_to_words_fr(x: float) -> str:
    text = num2words(x, lang='fr')
    return text.replace('virgule', 'dinars zéro')

# ---------- 1. Synthetic Data Generation ----------
F = Faker('fr_FR'); Faker.seed(42); random.seed(42)

def generate_valid_rib():
    def valid(v: str) -> bool:
        s = v.replace(' ', '').replace('-', '')
        if len(s) != 20:
            return False
        n = int(s[:-2] + '00')
        return 97 - (n % 97) == int(s[-2:])
    while True:
        base = ''.join(str(random.randint(0,9)) for _ in range(18))
        for i in range(100):
            cand = base + f"{i:02d}"
            if valid(cand):
                return cand


def generate_synthetic(n_legit=2500, n_fraud=500):
    rows = []
    fraud_patterns = [
        'missing_signature', 'invalid_barcode', 'invalid_rib',
        'amount_outlier', 'words_mismatch', 'date_violation', 'place_mismatch'
    ]
    for fraud in [False]*n_legit + [True]*n_fraud:
        amt = round(random.uniform(500,10000),3)
        words = amount_to_words_fr(amt)
        rib = generate_valid_rib()
        sig = True
        barcode_valid = True
        date_created = F.date_between(start_date='-1y', end_date='today')
        date_due = F.date_between(start_date=date_created, end_date='+6m')
        place_created = F.city()
        drawer = F.name()
        payer_addr = F.address().replace('\n',' ')
        if fraud:
            pat = random.choice(fraud_patterns)
            if pat == 'missing_signature': sig = False
            elif pat == 'invalid_barcode': barcode_valid = False
            elif pat == 'invalid_rib': rib = rib[:-2] + f"{random.randint(0,99):02d}"
            elif pat == 'amount_outlier': amt = round(random.uniform(20000,50000),3); words = amount_to_words_fr(amt)
            elif pat == 'words_mismatch': words += ' erreurs'
            elif pat == 'date_violation':
                if random.choice([True,False]): date_due = date_created - pd.Timedelta(days=1)
                else: date_due = date_created + pd.Timedelta(days=random.randint(365,730))
            elif pat == 'place_mismatch':
                other = F.city(); place_created = other
                payer_addr = payer_addr.replace(other, F.city())
        rows.append({
            'traite_num': str(F.random_number(digits=12)),
            'amount_digits': amt,
            'amount_words': words,
            'bank': F.company(),
            'rib': rib,
            'signature_detected': sig,
            'barcode_validates_traite': barcode_valid,
            'date_created': date_created.strftime('%Y-%m-%d'),
            'date_due': date_due.strftime('%Y-%m-%d'),
            'place_created': place_created,
            'drawer_name': drawer,
            'payer_name_address': payer_addr,
            'fraud_label': int(fraud)
        })
    df = pd.DataFrame(rows)
    df.to_sql('drafts', ENGINE, if_exists='replace', index=False)
    print("Synthetic data generated.")
    return df

# ---------- 2. Data Cleaning & Preprocessing ----------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['date_created'] = pd.to_datetime(df['date_created'], errors='coerce')
    df['date_due'] = pd.to_datetime(df['date_due'], errors='coerce')
    for col in ['bank','place_created','drawer_name','payer_name_address','amount_words']: df[col] = df[col].astype(str).str.strip()
    df['amount_digits'] = pd.to_numeric(df['amount_digits'], errors='coerce')
    df['signature_detected'] = df['signature_detected'].astype(bool)
    df['barcode_validates_traite'] = df['barcode_validates_traite'].astype(bool)
    return df.dropna(subset=['amount_digits','date_created','date_due','rib'])

# ---------- 3. Build/Refresh Statistics ----------
def build_stats():
    df = pd.read_sql('drafts', ENGINE)
    df = clean_data(df)
    pop = df['amount_digits']
    pop_stats = {'mean':pop.mean(),'std':pop.std(),'lo':pop.quantile(0.01),'hi':pop.quantile(0.99)}
    rib_stats = df.groupby('rib')['amount_digits'].agg(
        mean_amount='mean', std_amount='std', count='count',
        pct_1=lambda x: x.quantile(0.01), pct_99=lambda x: x.quantile(0.99)
    )
    bank_stats = df.groupby('bank')['amount_digits'].agg(
        mean='mean', std='std', lo=lambda x: x.quantile(0.01), hi=lambda x: x.quantile(0.99)
    )
    return pop_stats, rib_stats, bank_stats

# ---------- 4. Feature Engineering ----------
def featurize(df: pd.DataFrame, pop_stats, rib_stats, bank_stats,
              autoenc=None) -> pd.DataFrame:
    df = clean_data(df)
    feats=[]
    for _,r in df.iterrows():
        f={}; amt=r['amount_digits']; f['amount']=amt
        f['mismatch']=int(r['amount_words']!= amount_to_words_fr(amt))
        f['sig_missing']=int(not r['signature_detected']); f['barcode_bad']=int(not r['barcode_validates_traite'])
        f['rib_invalid']=int(not is_valid_rib(r['rib']))
        f['gap_days']=(r['date_due']-r['date_created']).days
        f['payer_len']=len(r['payer_name_address']);f['drawer_len']=len(r['drawer_name'])
        # z-score
        if r['rib'] in rib_stats.index and rib_stats.at[r['rib'],'count']>=5:
            s=rib_stats.loc[r['rib']];μ,σ=s['mean_amount'],s['std_amount']
        elif r['bank'] in bank_stats.index:
            s=bank_stats.loc[r['bank']];μ,σ=s['mean'],s['std']
        else:μ,σ=pop_stats['mean'],pop_stats['std']
        z=(amt-μ)/σ if σ>0 else 0;f['z_score']=z;f['amount_incompatible']=int(abs(z)>3)
        if autoenc:
            recon=autoenc.predict(np.array([[amt]]));f['anom_err']=float(abs(recon-amt))
        feats.append(f)
    return pd.DataFrame(feats)

# ---------- 5. Model Training & Saving ----------
def train(data_csv: str, model_dir: str):
    os.makedirs(model_dir,exist_ok=True)
    df=pd.read_csv(data_csv);df.to_sql('drafts',ENGINE,if_exists='replace',index=False)
    pop_stats,rib_stats,bank_stats=build_stats()
    # Autoencoder
    amounts=df[['amount_digits']].values.astype('float32')
    inp=layers.Input((1,));x=layers.Dense(8,activation='relu')(inp)
    x=layers.Dense(4,activation='relu')(x);x=layers.Dense(8,activation='relu')(x)
    out=layers.Dense(1)(x);autoenc=models.Model(inp,out)
    autoenc.compile('adam','mse');autoenc.fit(amounts,amounts,epochs=20,batch_size=32,validation_split=0.2,callbacks=[callbacks.EarlyStopping(patience=5)])
    autoenc.save(f"{model_dir}/autoencoder")
    # Classifier
    fea=featurize(df,pop_stats,rib_stats,bank_stats,autoenc)
    y=df['fraud_label'].values;X=fea.values.astype('float32')
    inpt=layers.Input((X.shape[1],));y1=layers.Dense(16,activation='relu')(inpt)
    y1=layers.Dense(8,activation='relu')(y1);y1=layers.Dense(1,activation='sigmoid')(y1)
    clf=models.Model(inpt,y1);clf.compile('adam','binary_crossentropy',['accuracy'])
    clf.fit(X,y,epochs=30,batch_size=32,validation_split=0.2,callbacks=[callbacks.EarlyStopping(patience=5)])
    clf.save(f"{model_dir}/classifier")
    joblib.dump({'pop':pop_stats,'rib':rib_stats,'bank':bank_stats},f"{model_dir}/stats.pkl")
    print(f"Models and stats saved to {model_dir}")

# ---------- 6. Inference ----------
def predict(ocr_json: dict, model_dir: str) -> dict:
    autoenc=tf.keras.models.load_model(f"{model_dir}/autoencoder")
    clf=tf.keras.models.load_model(f"{model_dir}/classifier")
    stats=joblib.load(f"{model_dir}/stats.pkl");pop_stats,rs,bs=stats['pop'],stats['rib'],stats['bank']
    df=pd.DataFrame([ocr_json]);df['amount_digits']=df['amount_digits'].astype(float)
    df['amount_words']=df['amount_words'];df['bank']=df['bank'];df['rib']=df['rib']
    df['signature_detected']=df['signature_detected'];df['barcode_validates_traite']=df['barcode_validates_traite']
    df['date_created']=pd.to_datetime(df['date_created']);df['date_due']=pd.to_datetime(df['date_due'])
    df['place_created']=df['place_created'];df['drawer_name']=df['drawer_name'];df['payer_name_address']=df['payer_name_address']
    fea=featurize(df,pop_stats,rs,bs,autoenc)
    X=fea.values.astype('float32');prob=float(clf.predict(X)[0,0])
    return {'fraud_score':prob,'fraud_label':prob>0.5}

# ---------- 7. Onboard New RIB ----------
def onboard(rib: str, amt: float):
    df=pd.read_sql('drafts',ENGINE)
    print(f"Before: {len(df)} rows")
    # user must insert new record separately
    pop_stats,rib_stats,bank_stats=build_stats()
    print(f"Stats refreshed, RIB count: {len(rib_stats)}")

# ---------- 8. Data Visualization ----------
def visualize_data():
    import matplotlib.pyplot as plt
    df=pd.read_sql('drafts',ENGINE)
    df=clean_data(df)
    # plots as earlier...
    plt.figure();df[df.fraud_label==0].amount_digits.hist(alpha=0.5,label='Legit');df[df.fraud_label==1].amount_digits.hist(alpha=0.5,label='Fraud');plt.legend();plt.title('Amount Dist')
    plt.show()

# ---------- 9. CLI Entrypoint ----------
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['generate','train','predict','onboard','visualize'])
    p.add_argument('--data');p.add_argument('--model-dir');p.add_argument('--ocr');p.add_argument('--rib');p.add_argument('--amt',type=float)
    args=p.parse_args()
    if args.command=='generate':generate_synthetic()
    elif args.command=='train':train(args.data,args.model_dir)
    elif args.command=='predict':print(predict(json.load(open(args.ocr)),args.model_dir))
    elif args.command=='onboard':onboard(args.rib,args.amt)
    elif args.command=='visualize':visualize_data()