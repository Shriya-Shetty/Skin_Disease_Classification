import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
import pandas as pd
import datetime
import os

# Load the model
model = YOLO("best.pt")  # Make sure the file is in the same directory

# Title
st.title("Skin Disease Detection from Camera")

# User input
name = st.text_input("Enter your name")
age = st.number_input("Enter your age", min_value=0, max_value=120, step=1)
date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Capture image from camera
capture = st.button("Take Photo from Camera")

if capture:
    cap = cv2.VideoCapture(0)
    st.info("Press 's' to capture image and 'q' to quit camera")

    captured = False
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to open camera")
            break

        cv2.imshow('Camera - Press s to capture', frame)
        key = cv2.waitKey(1)

        if key == ord('s'):
            captured = True
            img_path = "captured_image.jpg"
            cv2.imwrite(img_path, frame)
            break
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if captured:
        st.success("Image captured!")
        image = Image.open(img_path)
        st.image(image, caption="Captured Image")

        # Predict
        results = model.predict(img_path)
        prediction = results[0].probs.top1conf.item()
        label = results[0].names[results[0].probs.top1]

        st.subheader(f"Prediction: {label}")
        st.write(f"Confidence: {prediction:.2f}")

        # Save to CSV
        data = {
            "Name": [name],
            "Age": [age],
            "DateTime": [date],
            "Prediction": [label],
            "Confidence": [prediction]
        }

        df_new = pd.DataFrame(data)

        if os.path.exists("database.csv"):
            df_existing = pd.read_csv("database.csv")
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.to_csv("database.csv", index=False)
        else:
            df_new.to_csv("database.csv", index=False)

        st.success("Data saved to database.csv")
