from ultralytics import YOLO
import streamlit as st
from PIL import Image
import sys
import cv2

# Page layout
st.set_page_config(
  page_title="OAIX",
  page_icon="🤖" 
)

# Header
st.header("OAIX Detection")

# Side bar
st.sidebar.header("Model configurations")

# Choose Model: Detection / Segmentation / Pose Estimation
model_type = st.sidebar.radio("Task", ["Detection", "Segmentation","Pose Estimation"])

# Select Confidence Value
confidence_value = float(st.sidebar.slider("Select Model Confidence Value", 25, 40, 100))/100