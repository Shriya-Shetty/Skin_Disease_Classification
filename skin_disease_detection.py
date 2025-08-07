import streamlit as st
import pandas as pd
import datetime
import os
import tempfile
from PIL import Image
import numpy as np
import io

# Handle imports with error catching
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from ultralytics import YOLO
    import torch
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# Configure Streamlit page
st.set_page_config(page_title="Skin Disease Detection", layout="wide")

# --- Function Definitions ---
def load_model():
    """Load YOLO model with comprehensive error handling"""
    if not YOLO_AVAILABLE:
        return None
    
    try:
        model_path = "best.pt"
        if not os.path.exists(model_path):
            st.error(f"Model file '{model_path}' not found in current directory")
            st.info("Please ensure 'best.pt' is in the same directory as this script")
            return None
        
        model = YOLO(model_path)
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading YOLO model: {str(e)}")
        st.info("Common solutions:")
        st.write("1. Install required packages: `pip install ultralytics torch torchvision`")
        return None

def analyze_image(image, name, age, model):
    """Analyze the image using YOLO model"""
    if model is None:
        st.error("Model is not loaded. Cannot analyze image.")
        return

    with st.spinner("🔄 Analyzing image..."):
        try:
            # Make prediction directly from PIL Image
            results = model.predict(image, verbose=False)
            
            # Process results for both classification and detection models
            if len(results) > 0:
                result = results[0]
                if hasattr(result, 'probs') and result.probs is not None:
                    # Classification model results
                    prediction_conf = float(result.probs.top1conf.item())
                    label = result.names[result.probs.top1]
                elif hasattr(result, 'boxes') and len(result.boxes) > 0:
                    # Detection model results
                    confidences = result.boxes.conf.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy()
                    
                    best_idx = np.argmax(confidences)
                    prediction_conf = float(confidences[best_idx])
                    label = result.names[int(classes[best_idx])]
                else:
                    st.error("❌ No valid predictions found. Please try with a different image.")
                    return
                
                display_results(label, prediction_conf, name, age)
            else:
                st.error("❌ No predictions found. Please try with a different image.")
        
        except Exception as e:
            st.error(f"❌ Error during prediction: {str(e)}")

def display_results(label, confidence, name, age):
    """Display prediction results and save to database"""
    st.subheader("🎯 Analysis Results")
    
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        st.success(f"**Prediction:** {label}")
    
    with res_col2:
        st.info(f"**Confidence:** {confidence:.1%}")
    
    if confidence > 0.8:
        st.success("🟢 High confidence prediction")
    elif confidence > 0.6:
        st.warning("🟡 Moderate confidence prediction")
    else:
        st.error("🔴 Low confidence prediction")
        st.info("💡 Consider consulting a healthcare professional for accurate diagnosis")
    
    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_success = save_to_database(name, age, date, label, confidence)
    
    if save_success:
        st.balloons()

def save_to_database(name, age, date, prediction, confidence):
    """Save analysis results to CSV database"""
    try:
        data = {
            "Name": [name],
            "Age": [age], 
            "DateTime": [date],
            "Prediction": [prediction],
            "Confidence": [confidence]  # Store as a float
        }
        df_new = pd.DataFrame(data)
        
        csv_file = "database.csv"
        
        if os.path.exists(csv_file):
            df_existing = pd.read_csv(csv_file)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_combined = df_new
        
        df_combined.to_csv(csv_file, index=False)
        st.success("💾 Results saved to database!")
        
        with st.expander("📊 View Recent Records", expanded=True):
            st.dataframe(
                df_combined.tail(5).style.highlight_max(axis=0),
                use_container_width=True
            )
            if st.button("Delete Last Record", type="secondary"):
                delete_last_record()
                st.rerun()
        
        return True
        
    except Exception as e:
        st.error(f"❌ Error saving to database: {str(e)}")
        return False

def delete_last_record():
    """Delete the last entry from the CSV database"""
    csv_file = "database.csv"
    if os.path.exists(csv_file):
        try:
            df_existing = pd.read_csv(csv_file)
            if not df_existing.empty:
                df_new = df_existing.iloc[:-1] # Remove the last row
                df_new.to_csv(csv_file, index=False)
                st.warning("🗑️ Last record deleted from database.")
            else:
                st.info("The database is already empty.")
        except Exception as e:
            st.error(f"Error deleting record: {e}")

# Cache the model loading
@st.cache_resource
def get_model():
    return load_model()

# --- Main Application ---
with st.expander("🔧 System Status", expanded=False):
    st.write("**Dependency Check:**")
    st.write(f"- OpenCV (cv2): {'✅ Available' if CV2_AVAILABLE else '❌ Missing'}")
    st.write(f"- YOLO/PyTorch: {'✅ Available' if YOLO_AVAILABLE else '❌ Missing'}")
    
    if not YOLO_AVAILABLE:
        st.error("**Required Installation:**")
        st.code("pip install ultralytics torch torchvision", language="bash")

st.title("🔬 Skin Disease Detection App")

if not YOLO_AVAILABLE:
    st.stop()

model = get_model()
if model is None:
    st.stop()

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("👤 Patient Information")
    name = st.text_input("Enter your name", placeholder="John Doe")
    age = st.number_input("Enter your age", min_value=0, max_value=120, step=1, value=25)
    
    st.subheader("📷 Image Input")
    uploaded_file = st.file_uploader(
        "Choose an image file...", 
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
        help="Upload a clear image of the skin area to be analyzed"
    )
    if CV2_AVAILABLE:
        st.caption("Or use your camera:")
        camera_file = st.camera_input("Take a picture")

with col2:
    image_to_analyze = None
    if uploaded_file is not None:
        try:
            image_to_analyze = Image.open(uploaded_file)
            st.image(image_to_analyze, caption="Uploaded Image", use_container_width=True)
        except Exception as e:
            st.error(f"Error loading uploaded image: {str(e)}")
    
    if camera_file is not None:
        try:
            image_to_analyze = Image.open(camera_file)
            st.image(image_to_analyze, caption="Captured Image", use_container_width=True)
        except Exception as e:
            st.error(f"Error loading captured image: {str(e)}")
    
    if image_to_analyze is not None:
        if not name.strip():
            st.warning("⚠️ Please enter your name before analysis")
        else:
            if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                analyze_image(image_to_analyze, name, age, model)

with st.sidebar:
    st.header("ℹ️ About")
    st.info("AI-powered skin disease detection using computer vision.")
    
    st.warning("⚠️ **Medical Disclaimer**")
    st.write("This tool is for educational and screening purposes only. Always consult healthcare professionals for proper medical diagnosis and treatment.")
    
    st.header("📋 Instructions")
    st.write("1. Enter your personal information")
    st.write("2. Upload an image or use camera")
    st.write("3. Click 'Analyze' for AI prediction")
    st.write("4. Results are saved automatically")
    
    st.header("📊 Database Stats")
    try:
        if os.path.exists("database.csv"):
            df = pd.read_csv("database.csv")
            st.metric("Total Analyses", len(df))
            
            if len(df) > 0:
                most_common = df['Prediction'].mode()
                if not most_common.empty:
                    st.metric("Most Common", most_common.iloc[0])
                
                try:
                    avg_conf = pd.to_numeric(df['Confidence'], errors='coerce').mean()
                    if not pd.isna(avg_conf):
                        st.metric("Avg Confidence", f"{avg_conf:.1%}")
                except Exception:
                    pass
        else:
            st.metric("Total Analyses", 0)
    except Exception:
        st.metric("Total Analyses", "Error loading")

st.markdown("---")
st.markdown("**📝 Note:** For best results, ensure good lighting and clear focus on the skin area to be analyzed.")
