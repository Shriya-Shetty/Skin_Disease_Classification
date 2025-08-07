import streamlit as st
import pandas as pd
import datetime
import os
import tempfile
from PIL import Image
import numpy as np

# Handle imports with error catching
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    st.warning("OpenCV (cv2) not available. Camera capture will be disabled.")

try:
    from ultralytics import YOLO
    import torch
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    st.error("YOLO/PyTorch not available. Please install required packages.")

# Configure Streamlit page
st.set_page_config(page_title="Skin Disease Detection", layout="wide")

# Function definitions
def load_model():
    """Load YOLO model with comprehensive error handling"""
    if not YOLO_AVAILABLE:
        return None
    
    try:
        # Check if model file exists
        model_path = "best.pt"
        if not os.path.exists(model_path):
            st.error(f"Model file '{model_path}' not found in current directory")
            st.info("Please ensure 'best.pt' is in the same directory as this script")
            return None
        
        # Load model with explicit error handling
        model = YOLO(model_path)
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading YOLO model: {str(e)}")
        st.info("Common solutions:")
        st.write("1. Install required packages: `pip install ultralytics torch torchvision`")
        st.write("2. Ensure 'best.pt' model file is present")
        st.write("3. Check if your model file is corrupted")
        return None

def capture_from_camera():
    """Capture image from camera with improved error handling"""
    try:
        cap = cv2.VideoCapture(0)
        
        # Set camera properties for better quality
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not cap.isOpened():
            st.error("❌ Cannot access camera. Please check:")
            st.write("- Camera is connected and functional")
            st.write("- No other application is using the camera")
            st.write("- Camera permissions are granted")
            return None
        
        st.info("📹 Camera opened! Press 'S' to capture, 'Q' to quit")
        
        captured_frame = None
        frame_count = 0
        max_frames = 1500  # 50 seconds at 30fps
        
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to read from camera")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Display instructions on frame
            cv2.putText(frame, "Press 'S' to capture, 'Q' to quit", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Skin Disease Detection - Camera', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s') or key == ord('S'):
                captured_frame = frame.copy()
                st.success("📸 Image captured successfully!")
                break
            elif key == ord('q') or key == ord('Q'):
                st.info("Camera closed by user")
                break
            
            frame_count += 1
        
        cap.release()
        cv2.destroyAllWindows()
        
        if captured_frame is not None:
            # Convert BGR to RGB for PIL
            captured_frame_rgb = cv2.cvtColor(captured_frame, cv2.COLOR_BGR2RGB)
            return Image.fromarray(captured_frame_rgb)
        
        return None
        
    except Exception as e:
        st.error(f"Camera error: {str(e)}")
        return None

def analyze_image(image, name, age, model):
    """Analyze the image using YOLO model"""
    with st.spinner("🔄 Analyzing image..."):
        try:
            # Save image temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                image.save(tmp_file.name, format='JPEG')
                img_path = tmp_file.name
            
            # Make prediction
            results = model.predict(img_path, verbose=False)
            
            # Process results
            if len(results) > 0 and hasattr(results[0], 'probs') and results[0].probs is not None:
                # Classification model results
                prediction_conf = float(results[0].probs.top1conf.item())
                label = results[0].names[results[0].probs.top1]
                
                display_results(label, prediction_conf, name, age)
                
            elif len(results) > 0 and hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
                # Detection model results
                # Get the highest confidence detection
                confidences = results[0].boxes.conf.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy()
                
                best_idx = np.argmax(confidences)
                prediction_conf = float(confidences[best_idx])
                label = results[0].names[int(classes[best_idx])]
                
                display_results(label, prediction_conf, name, age)
                
            else:
                st.error("❌ No valid predictions found. Please try with a different image.")
                st.info("Tips for better results:")
                st.write("- Ensure good lighting")
                st.write("- Focus on the skin area")
                st.write("- Avoid blurry images")
            
        except Exception as e:
            st.error(f"❌ Error during prediction: {str(e)}")
            st.info("This might be due to:")
            st.write("- Incompatible image format")
            st.write("- Model compatibility issues")
            st.write("- Insufficient system resources")
        
        finally:
            # Clean up temporary file
            try:
                if 'img_path' in locals() and os.path.exists(img_path):
                    os.unlink(img_path)
            except:
                pass

def display_results(label, confidence, name, age):
    """Display prediction results and save to database"""
    st.subheader("🎯 Analysis Results")
    
    # Create results columns
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        st.success(f"**Prediction:** {label}")
    
    with res_col2:
        st.info(f"**Confidence:** {confidence:.1%}")
    
    # Confidence interpretation
    if confidence > 0.8:
        st.success("🟢 High confidence prediction")
    elif confidence > 0.6:
        st.warning("🟡 Moderate confidence prediction")
    else:
        st.error("🔴 Low confidence prediction")
        st.info("💡 Consider consulting a healthcare professional for accurate diagnosis")
    
    # Save to database
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
            "Confidence": [f"{confidence:.3f}"]
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
        
        # Show recent entries
        with st.expander("📊 View Recent Records"):
            st.dataframe(
                df_combined.tail(5).style.highlight_max(axis=0),
                use_container_width=True
            )
        
        return True
        
    except Exception as e:
        st.error(f"❌ Error saving to database: {str(e)}")
        return False

# Cache the model loading
@st.cache_resource
def get_model():
    return load_model()

# Main Application
# Display dependency status
with st.expander("🔧 System Status", expanded=False):
    st.write("**Dependency Check:**")
    st.write(f"- OpenCV (cv2): {'✅ Available' if CV2_AVAILABLE else '❌ Missing'}")
    st.write(f"- YOLO/PyTorch: {'✅ Available' if YOLO_AVAILABLE else '❌ Missing'}")
    
    if not YOLO_AVAILABLE:
        st.error("**Required Installation:**")
        st.code("pip install ultralytics torch torchvision opencv-python", language="bash")

# Title
st.title("🔬 Skin Disease Detection App")

# Only proceed if dependencies are available
if not YOLO_AVAILABLE:
    st.stop()

# Load model
model = get_model()
if model is None:
    st.stop()

# Create columns for better layout
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("👤 Patient Information")
    name = st.text_input("Enter your name", placeholder="John Doe")
    age = st.number_input("Enter your age", min_value=0, max_value=120, step=1, value=25)
    
    # Input method selection
    st.subheader("📷 Image Input")
    if CV2_AVAILABLE:
        input_method = st.radio("Choose input method:", ["Upload Image", "Camera Capture"])
    else:
        input_method = "Upload Image"
        st.info("Camera capture disabled - OpenCV not available")

with col2:
    if input_method == "Upload Image":
        st.subheader("📁 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image file...", 
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            help="Upload a clear image of the skin area to be analyzed"
        )
        
        if uploaded_file is not None:
            try:
                # Display uploaded image
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_container_width=True)
                
                # Check if name is provided
                if not name.strip():
                    st.warning("⚠️ Please enter your name before analysis")
                else:
                    # Analyze button
                    if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                        analyze_image(image, name, age, model)
                        
            except Exception as e:
                st.error(f"Error loading image: {str(e)}")
    
    elif input_method == "Camera Capture" and CV2_AVAILABLE:
        st.subheader("📷 Camera Capture")
        
        if not name.strip():
            st.warning("⚠️ Please enter your name before using camera")
        else:
            col_cam1, col_cam2 = st.columns(2)
            
            with col_cam1:
                if st.button("📸 Open Camera", type="primary", use_container_width=True):
                    captured_image = capture_from_camera()
                    if captured_image is not None:
                        st.session_state['captured_image'] = captured_image
            
            with col_cam2:
                if st.button("🔄 Clear Capture", use_container_width=True):
                    if 'captured_image' in st.session_state:
                        del st.session_state['captured_image']
                    st.rerun()
            
            # Display captured image if available
            if 'captured_image' in st.session_state:
                st.image(st.session_state['captured_image'], caption="Captured Image", use_container_width=True)
                
                if st.button("🔍 Analyze Captured Image", type="primary", use_container_width=True):
                    analyze_image(st.session_state['captured_image'], name, age, model)

# Sidebar information
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
                # Most common prediction
                most_common = df['Prediction'].mode()
                if not most_common.empty:
                    st.metric("Most Common", most_common.iloc[0])
                
                # Average confidence
                try:
                    avg_conf = pd.to_numeric(df['Confidence'], errors='coerce').mean()
                    if not pd.isna(avg_conf):
                        st.metric("Avg Confidence", f"{avg_conf:.1%}")
                except:
                    pass
        else:
            st.metric("Total Analyses", 0)
    except Exception:
        st.metric("Total Analyses", "Error loading")

# Footer
st.markdown("---")
st.markdown("**📝 Note:** For best results, ensure good lighting and clear focus on the skin area to be analyzed.")
