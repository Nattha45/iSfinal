import streamlit as st
import joblib
import numpy as np

# โหลดโมเดล
model = joblib.load('dog_breed_model.pkl')

# โหลด LabelEncoders จากไฟล์เดียว
label_encoders = joblib.load('label_encoders.pkl')
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