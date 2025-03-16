import os
import streamlit as st
import joblib
import numpy as np

# หาพาธของโฟลเดอร์ที่ไฟล์ `app.py` อยู่
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# กำหนดพาธของไฟล์โมเดลและ encoders
MODEL_PATH = os.path.join(BASE_DIR, "dog_breed_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "label_encoders.pkl")

# ตรวจสอบว่าไฟล์โมเดลมีอยู่จริง
if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model file not found: {MODEL_PATH}")
    st.stop()

if not os.path.exists(ENCODER_PATH):
    st.error(f"❌ Encoder file not found: {ENCODER_PATH}")
    st.stop()

# โหลดโมเดล
model = joblib.load(MODEL_PATH)

# โหลด LabelEncoders
label_encoders = joblib.load(ENCODER_PATH)
label_encoder_breed = label_encoders['breed']
label_encoder_traits = label_encoders['traits']

# กำหนดหัวข้อและคำอธิบาย
st.title("Dog Breed Character Traits Predictor")
st.write("Enter a dog breed to predict its character traits.")

# ช่องป้อนข้อมูล
breed_input = st.text_input("Enter a dog breed (e.g., Labrador):", "").strip()

# ปุ่มทำนาย
if st.button("Predict"):
    if not breed_input:
        st.error("Please enter a breed.")
    else:
        if breed_input not in label_encoder_breed.classes_:
            st.warning(f"Breed '{breed_input}' not found in the dataset.")
        else:
            try:
                breed_encoded = label_encoder_breed.transform([breed_input]).reshape(1, -1)
                traits_encoded = model.predict(breed_encoded)
                traits = label_encoder_traits.inverse_transform(traits_encoded)[0]
                st.success(f"Predicted Character Traits for {breed_input}: {traits}")
            except Exception as e:
                st.error(f"Error: {str(e)}")

# ข้อมูลเพิ่มเติม
st.write("Powered by Streamlit and Decision Tree Classifier")
