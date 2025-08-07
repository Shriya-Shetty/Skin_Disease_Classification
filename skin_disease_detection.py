import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
import pandas as pd
import datetime
import os
import tempfile

# Configure Streamlit page
st.set_page_config(page_title="Skin Disease Detection", layout="wide")

# Load the model with error handling
@st.cache_resource
def load_model():
    try:
        model = YOLO("best.pt")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Make sure 'best.pt' is in the same directory as this script")
        return None

model = load_model()

# Title
st.title("🔬 Skin Disease Detection from Camera")

# Check if model loaded successfully
if model is None:
    st.stop()

# Create columns for better layout
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Patient Information")
    # User input
    name = st.text_input("Enter your name", placeholder="John Doe")
    age = st.number_input("Enter your age", min_value=0, max_value=120, step=1, value=25)
    
    # Alternative: File upload instead of camera (more reliable in web apps)
    st.subheader("Image Input")
    input_method = st.radio("Choose input method:", ["Upload Image", "Camera Capture"])

with col2:
    if input_method == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None and name.strip():
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                image.save(tmp_file.name)
                img_path = tmp_file.name
            
            # Predict button
            if st.button("🔍 Analyze Image", type="primary"):
                with st.spinner("Analyzing image..."):
                    try:
                        # Make prediction
                        results = model.predict(img_path, verbose=False)
                        
                        # Check if results contain predictions
                        if len(results) > 0 and hasattr(results[0], 'probs') and results[0].probs is not None:
                            prediction_conf = float(results[0].probs.top1conf.item())
                            label = results[0].names[results[0].probs.top1]
                            
                            # Display results
                            st.subheader("🎯 Analysis Results")
                            st.success(f"**Prediction:** {label}")
                            st.info(f"**Confidence:** {prediction_conf:.2%}")
                            
                            # Confidence interpretation
                            if prediction_conf > 0.8:
                                st.success("High confidence prediction")
                            elif prediction_conf > 0.6:
                                st.warning("Moderate confidence prediction")
                            else:
                                st.error("Low confidence prediction - consider consulting a medical professional")
                            
                            # Save to database
                            date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            save_to_database(name, age, date, label, prediction_conf)
                            
                        else:
                            st.error("No valid predictions found. Please try with a different image.")
                    
                    except Exception as e:
                        st.error(f"Error during prediction: {e}")
                    
                    finally:
                        # Clean up temporary file
                        if os.path.exists(img_path):
                            os.unlink(img_path)
    
    else:  # Camera Capture
        st.info("📸 Camera capture functionality")
        st.warning("Note: Camera capture works better in desktop applications. For web deployment, consider using the upload option.")
        
        if st.button("📷 Open Camera", type="primary") and name.strip():
            try:
                # Camera capture with better error handling
                cap = cv2.VideoCapture(0)
                
                if not cap.isOpened():
                    st.error("❌ Could not open camera. Please check if:")
                    st.write("- Camera is connected and working")
                    st.write("- No other application is using the camera")
                    st.write("- Camera permissions are granted")
                else:
                    st.info("📹 Camera opened successfully!")
                    st.info("Instructions: Press 's' to capture image, 'q' to quit")
                    
                    captured = False
                    img_path = "captured_image.jpg"
                    
                    # Add timeout to prevent infinite loop
                    frame_count = 0
                    max_frames = 3000  # ~100 seconds at 30fps
                    
                    while cap.isOpened() and frame_count < max_frames:
                        ret, frame = cap.read()
                        if not ret:
                            st.error("Failed to read frame from camera")
                            break
                        
                        # Display frame
                        cv2.imshow('Skin Disease Detection - Press S to capture, Q to quit', frame)
                        
                        key = cv2.waitKey(1) & 0xFF
                        
                        if key == ord('s') or key == ord('S'):
                            captured = True
                            cv2.imwrite(img_path, frame)
                            st.success("📸 Image captured successfully!")
                            break
                        elif key == ord('q') or key == ord('Q'):
                            st.info("Camera closed by user")
                            break
                        
                        frame_count += 1
                    
                    cap.release()
                    cv2.destroyAllWindows()
                    
                    if captured and os.path.exists(img_path):
                        # Display captured image
                        image = Image.open(img_path)
                        st.image(image, caption="Captured Image", use_column_width=True)
                        
                        # Make prediction
                        with st.spinner("Analyzing captured image..."):
                            try:
                                results = model.predict(img_path, verbose=False)
                                
                                if len(results) > 0 and hasattr(results[0], 'probs') and results[0].probs is not None:
                                    prediction_conf = float(results[0].probs.top1conf.item())
                                    label = results[0].names[results[0].probs.top1]
                                    
                                    # Display results
                                    st.subheader("🎯 Analysis Results")
                                    st.success(f"**Prediction:** {label}")
                                    st.info(f"**Confidence:** {prediction_conf:.2%}")
                                    
                                    # Save to database
                                    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    save_to_database(name, age, date, label, prediction_conf)
                                else:
                                    st.error("No valid predictions found.")
                            
                            except Exception as e:
                                st.error(f"Error during prediction: {e}")
                        
                        # Clean up
                        try:
                            os.remove(img_path)
                        except:
                            pass
            
            except Exception as e:
                st.error(f"Camera error: {e}")

# Function to save data to database
def save_to_database(name, age, date, prediction, confidence):
    try:
        data = {
            "Name": [name],
            "Age": [age],
            "DateTime": [date],
            "Prediction": [prediction],
            "Confidence": [confidence]
        }
        df_new = pd.DataFrame(data)
        
        csv_file = "database.csv"
        
        if os.path.exists(csv_file):
            df_existing = pd.read_csv(csv_file)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_combined = df_new
        
        df_combined.to_csv(csv_file, index=False)
        st.success("💾 Data saved to database successfully!")
        
        # Show recent entries
        with st.expander("📋 Recent Database Entries"):
            st.dataframe(df_combined.tail(5))
            
    except Exception as e:
        st.error(f"Error saving to database: {e}")

# Sidebar with additional information
with st.sidebar:
    st.header("ℹ️ Information")
    st.info("This application uses AI to detect skin conditions from images.")
    st.warning("⚠️ **Disclaimer:** This tool is for educational purposes only and should not replace professional medical diagnosis.")
    
    st.header("📊 Usage Statistics")
    if os.path.exists("database.csv"):
        df = pd.read_csv("database.csv")
        st.metric("Total Scans", len(df))
        if len(df) > 0:
            most_common = df['Prediction'].mode().iloc[0] if not df['Prediction'].mode().empty else "N/A"
            st.metric("Most Common Prediction", most_common)
    else:
        st.metric("Total Scans", 0)

# Footer
st.markdown("---")
st.markdown("**Note:** Please ensure you have proper lighting and the skin area is clearly visible for best results.")
