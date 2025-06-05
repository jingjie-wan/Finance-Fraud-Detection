from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import pandas as pd
import numpy as np
import re
import easyocr
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import uvicorn
import time

# ---------- Model and Encoders Setup ----------
dummy_data = pd.DataFrame({
    "file_name": [f"batch1-000{i}.jpg" for i in range(10)],
    "Date": pd.date_range("2013-01-01", periods=10, freq="Y").strftime("%m/%d/%Y"),
    "Amount": np.round(np.random.uniform(500, 10000, size=10), 2),
    "invoice_no": np.random.randint(10000000, 99999999, size=10).astype(str),
    "Client": [f"Client {i}" for i in range(10)],
    "Client Address": [f"123 Main St Apt {i}, City {i}, ZZ {10000+i}" for i in range(10)],
    "Client Tax ID": [f"{np.random.randint(100,999)}-{np.random.randint(10,99)}-{np.random.randint(1000,9999)}" for _ in range(10)],
    "IBAN": [f"GB{np.random.randint(10, 99)}BANK{np.random.randint(100000, 999999)}{np.random.randint(10000000, 99999900)}" for _ in range(10)],
    "is_fraud": np.random.choice([0, 1], size=10, p=[0.7, 0.3])
})

# Feature engineering
le_tax = LabelEncoder()
le_iban = LabelEncoder()
dummy_data['Amount'] = dummy_data['Amount'].astype(float)
dummy_data['Date'] = pd.to_datetime(dummy_data['Date']).dt.dayofweek
dummy_data['Client Address Length'] = dummy_data['Client Address'].apply(len)
dummy_data['Client Tax ID Encoded'] = le_tax.fit_transform(dummy_data['Client Tax ID'])
dummy_data['IBAN Encoded'] = le_iban.fit_transform(dummy_data['IBAN'])

X = dummy_data[['Date', 'Amount', 'Client Address Length', 'Client Tax ID Encoded', 'IBAN Encoded']]
y = dummy_data['is_fraud']

model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# ---------- FastAPI Setup ----------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="."), name="static")
reader = easyocr.Reader(['en'])

# ---------- Prediction Logic ----------
def predict_invoice_fraud(image_path):
    start_time = time.time()

    results = reader.readtext(image_path)
    all_texts = [text for (_, text, _) in results]
    joined_text = " ".join(all_texts)

    date_match = re.search(r'(\d{2}[\/\.\-]\d{2}[\/\.\-]\d{4}|\d{4}[\/\.\-]\d{2}[\/\.\-]\d{2})', joined_text)
    amount_match = re.search(r'(Total|Amount\s*due|Paid)[^\d\$\u20AC\u20BA]*([\$\u20AC\u20BA]?\s?[\d\s]{1,10}[.,]\d{2})', joined_text)
    client_block = re.search(r'Client[:\-]?\s*(.+?)Tax\s*Id', joined_text, re.IGNORECASE)
    client_tax = re.search(r'Client.*?Tax\s*Id[:\-]?\s*([\d\-]+)', joined_text)
    iban_match = re.search(r'IBAN[:\-]?\s*([A-Z]{2}[0-9A-Z]{13,32})', joined_text)

    client_info = client_block.group(1).strip() if client_block else ''
    client_lines = re.split(r'\s(?=\d{5})', client_info)
    address_text = client_lines[1].strip() if len(client_lines) > 1 else client_info

    date_val = pd.to_datetime(date_match.group(0), errors='coerce').dayofweek if date_match else 0
    amount_val = float(amount_match.group(2).replace(" ", "").replace(",", ".").replace("$", "")) if amount_match else 0.0
    addr_len = len(address_text)
    tax_id_val = client_tax.group(1) if client_tax else 'UNKNOWN'
    iban_val = iban_match.group(1) if iban_match else 'UNKNOWN'

    tax_encoded = le_tax.transform([tax_id_val])[0] if tax_id_val in le_tax.classes_ else 0
    iban_encoded = le_iban.transform([iban_val])[0] if iban_val in le_iban.classes_ else 0

    X_input = pd.DataFrame([{ "Date": date_val, "Amount": amount_val, "Client Address Length": addr_len, "Client Tax ID Encoded": tax_encoded, "IBAN Encoded": iban_encoded }])
    pred = model.predict(X_input)[0]

    elapsed = round(time.time() - start_time, 2)
    print(f"[DEBUG] INPUT: date={date_val}, amount={amount_val}, address_len={addr_len}, tax_id={tax_id_val}, iban={iban_val}")
    return ("FRAUD" if pred == 1 else "NOT FRAUD"), elapsed

# ---------- Routes ----------
@app.get("/", response_class=HTMLResponse)
def main():
    return """
    <html lang="en">
        <head>
            <title>Invoice Fraud Detection</title>
            <style>
                body { font-family: Arial, sans-serif; background-color: #f4f4f4; padding: 40px; text-align: center; }
                form { background: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); display: inline-block; }
                h2 { color: #333; }
                input[type=file] { margin: 20px 0; padding: 10px; }
                input[type=submit] { background-color: #28a745; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
                input[type=submit]:hover { background-color: #218838; }
                #loading { display: none; margin-top: 20px; font-weight: bold; color: #555; }
                .spinner { display: inline-block; width: 24px; height: 24px; border: 3px solid #ccc; border-top: 3px solid #333; border-radius: 50%; animation: spin 1s linear infinite; }
                @keyframes spin { 100% { transform: rotate(360deg); } }
            </style>
            <script>
                function showLoading() {
                    document.getElementById("loading").style.display = "block";
                }
            </script>
        </head>
        <body>
            <h2>Upload an Invoice Image</h2>
            <form action="/predict" enctype="multipart/form-data" method="post" onsubmit="showLoading()">
                <input name="file" type="file" accept="image/*">
                <br>
                <input type="submit" value="Check for Fraud">
            </form>
            <div id="loading"><div class="spinner"></div> Processing image, please wait...</div>
        </body>
    </html>
    """

@app.post("/predict", response_class=HTMLResponse)
def predict(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    prediction, duration = predict_invoice_fraud(temp_path)
    os.remove(temp_path)

    color = "#dc3545" if prediction == "FRAUD" else "#28a745"
    return f"""
    <html lang="en">
        <head>
            <title>Prediction Result</title>
            <style>
                body {{ font-family: Arial, sans-serif; background-color: #f4f4f4; padding: 40px; text-align: center; }}
                .result {{ background-color: {color}; color: white; padding: 20px; border-radius: 10px; display: inline-block; font-size: 24px; }}
                a {{ display: inline-block; margin-top: 20px; text-decoration: none; color: #007bff; }}
            </style>
        </head>
        <body>
            <div class="result">Prediction: <strong>{prediction}</strong><br>Processed in {duration} seconds</div><br>
            <a href="/">Go Back</a>
        </body>
    </html>
    """

#if __name__ == "__main__":
#    uvicorn.run(app, host="0.0.0.0", port=8000)

