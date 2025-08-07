import streamlit as st
import torch
import onnxruntime as ort
from PIL import Image
import numpy as np
import os

# Set up class labels (example: replace with your real labels)
CLASS_NAMES = ['Acne', 'Eczema', 'Melanoma', 'Psoriasis', 'Healthy']

def load_pt_model(model_path):
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()
    return model

def load_onnx_session(model_path):
    return ort.InferenceSession(model_path)

def preprocess_image(image):
    image = image.convert('RGB')
    image = image.resize((224, 224))
    image_array = np.array(image).astype('float32') / 255.0
    image_array = image_array.transpose(2, 0, 1)  # HWC → CHW
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def predict_with_pt(model, image_tensor):
    with torch.no_grad():
        image_tensor = torch.tensor(image_tensor)
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs.data, 1)
        return CLASS_NAMES[predicted.item()]

def predict_with_onnx(session, image_tensor):
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: image_tensor})
    predicted_class = np.argmax(outputs[0])
    return CLASS_NAMES[predicted_class]

# Streamlit UI
st.set_page_config(page_title="Skin Disease Classifier", layout="centered")
st.title("🧴 Skin Disease Classification App")

model_file = st.file_uploader("Upload Model (.pt or .onnx)", type=['pt', 'onnx'])

if model_file:
    model_ext = os.path.splitext(model_file.name)[1]

    with open(f"models/{model_file.name}", "wb") as f:
        f.write(model_file.read())

    st.success(f"Model `{model_file.name}` uploaded.")

    image_file = st.file_uploader("Upload a Skin Image", type=['jpg', 'jpeg', 'png'])

    if image_file:
        image = Image.open(image_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        img_tensor = preprocess_image(image)

        if model_ext == '.pt':
            model = load_pt_model(f"models/{model_file.name}")
            prediction = predict_with_pt(model, img_tensor)
        elif model_ext == '.onnx':
            session = load_onnx_session(f"models/{model_file.name}")
            prediction = predict_with_onnx(session, img_tensor)
        else:
            st.error("Unsupported model format.")
            st.stop()

        st.subheader("🩺 Prediction:")
        st.success(f"**{prediction}**")

