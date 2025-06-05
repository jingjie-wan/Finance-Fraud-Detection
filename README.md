# How to run:
Download fraud_detection.py. Start the app with:
```
uvicorn rag_ai_agent:app --host 0.0.0.0 --port 8082 --reload
```
in your terminal. Then open your browser at http://127.0.0.1:8000 to upload images and see predictions.

# Data
For information security reasons, I used dummy data (public invoice images) on Kaggle: https://www.kaggle.com/datasets/osamahosamabdellatif/high-quality-invoice-images-for-ocr/data for training and testing.

Data Example:

![image](https://github.com/user-attachments/assets/31c2839d-6050-44e9-88c0-c4780c2d483b)

# Input and Output
## Input
A .jpg (invoice) image you want to test out.

![image](https://github.com/user-attachments/assets/4c8affa3-f78d-49a3-91da-6efc7e1904f0)



## Output
Return result of either 'FRUAD' or 'NOT FRAUD'.

![image](https://github.com/user-attachments/assets/f1084312-6cf4-4730-b612-64ce86ade2c8)

