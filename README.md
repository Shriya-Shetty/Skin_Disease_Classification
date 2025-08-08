# 🧠 Skin Disease Classification App using YOLOv8 + Streamlit

A full-featured web application that classifies various skin diseases using a custom-trained **YOLOv8** classification model. Users can either upload an image or use their webcam for real-time predictions.

The app is fully developed in **Streamlit**, supports **database integration**, and is **deployed on Streamlit Cloud** for easy access.

---

## 🔗 Live Demo

👉 [https://skindiseaseclassification.streamlit.app/](https://skindiseaseclassification.streamlit.app/)

---

## 🎯 Project Objective

Skin diseases can often be misdiagnosed without professional expertise. This project aims to build an accessible, fast, and lightweight AI-powered app that provides **instant predictions** for common skin diseases.

- ✅ Real-time skin disease classification
- ✅ Easy image upload or webcam capture
- ✅ Model trained on real-world dermatology dataset
- ✅ Results stored in a database

---

## 🧠 Model Details

| Attribute       | Value                     |
|----------------|---------------------------|
| Model Type      | YOLOv8n-cls               |
| Framework       | [Ultralytics YOLOv8](https://docs.ultralytics.com/) |
| Training Data   | [Kaggle Skin Disease Dataset](https://www.kaggle.com/datasets/subirbiswas19/skin-disease-dataset) |
| Input Size      | 224 × 224                 |
| Output          | Top-1 class + confidence  |
| Training Epochs | 25                        |
| File            | `best.pt` (exported YOLOv8 model) |


<img width="623" height="624" alt="image" src="https://github.com/user-attachments/assets/61125ac4-5500-4364-b42f-eef29356431f" />
<img width="829" height="630" alt="image" src="https://github.com/user-attachments/assets/1fade201-64d5-4de0-b3c3-6eae7544489b" />



