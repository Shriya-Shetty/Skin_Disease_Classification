import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.title("🧠 Skin Disease Classifier")

# Load the YOLO model
model = YOLO("best.pt")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Run prediction
    results = model(image)

    # Show class prediction
    for r in results:
        probs = r.probs
        if probs is not None:
            predicted_class = r.names[probs.top1]
            confidence = probs.top1conf
            st.success(f"Predicted: **{predicted_class}** ({confidence:.2%})")
        else:
            st.warning("No prediction found")
