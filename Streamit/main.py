from ultralytics import YOLO
import streamlit as st
from pathlib import Path
from PIL import Image
import sys
import cv2

# Page layout
st.set_page_config(
  page_title="OAIX",
  page_icon="🤖" 
)

# Get the absolute path of the current file
FILE = Path(__file__).resolve()

# Get the root directory of the current file
ROOT = FILE.parent

# Add the root path to the sys.path list
if ROOT not in sys.path:
  sys.path.append(str(ROOT))

# Get the relative path to the root directory with respect to the current working directory
ROOT = ROOT.relative_to(Path.cwd())

# Sources 
IMAGE = "Image"
VIDEO = "Video"

SOURCE_LIST = [IMAGE, VIDEO]

# Image Config
IMAGES_DIR = ROOT/'images'
DEFAULT_IMAGE = IMAGES_DIR/'image1.jpg'
DEFAULT_DETECT_IMAGE = IMAGES_DIR/'detectedimage1.jpg'

# Video Config
VIDEOS_DIR = ROOT/'videos'
VIDEOS_DICT = {
  'video 1':VIDEOS_DIR/'video1.mp4',
  'video 2':VIDEOS_DIR/'video2.mp4',
  'video 3':VIDEOS_DIR/'Esther_playing_football.mp4'
}

# Model Configuration
MODEL_DIR = ROOT/'weights'
DETECTION_MODEL = MODEL_DIR/'yolo11n.pt'
SEGMENTATION_MODEL = MODEL_DIR/'yolo11n-seg.pt'
POSE_MODEL = MODEL_DIR/'yolo11n-pose.pt'

# For the custom models
# DETECTION_MODE = ROOT/'custom_model_weight.pt'

# Header
st.header("OAIX Detection")

# Side bar
st.sidebar.header("Model configurations")

# Choose Model: Detection / Segmentation / Pose Estimation
model_type = st.sidebar.radio("Task", ["Detection", "Segmentation","Pose Estimation"])

# Select Confidence Value
confidence_value = float(st.sidebar.slider("Select Model Confidence Value", 25, 100, 40))/100

# Select Detection, Segmentation, Pose Estimation model
if model_type == "Detection":
  model_path = Path(DETECTION_MODEL)
elif model_type == "Segmentation":
  model_path = Path(SEGMENTATION_MODEL)
elif model_type == "Pose Estimation":
  model_path = Path(POSE_MODEL)

# Load the model
try:
  model = YOLO(model_path)
except Exception as e:
  st.error(f"Err: Unable to load model {model_path}")
  st.error(e)

# Image / Video Configuration
  
st.sidebar.header("Image/Video Config")
source_radio = st.sidebar.radio(
  "Select Source", SOURCE_LIST
)

source_image = None
if source_radio == IMAGE:
  source_image = st.sidebar.file_uploader(
    "Choose an Image ...", type = ("jpg","jpeg", "png", "bmp", "webp")
  )
  col1, col2 = st.columns(2)
  with col1:
    try:
      if source_image is None:
        default_image_path = str(DEFAULT_IMAGE)
        default_image = Image.open(default_image_path)
        st.image(default_image_path, caption="default image")
      else:
        uploaded_image = Image.open(source_image)
        st.image(uploaded_image, caption="uploaded Image")
    except Exception as e:
      st.error(e)
  with col2:
    try:
      if source_image is None:
        default_detected_image_path = str(DEFAULT_DETECT_IMAGE)
        default_detected_image = Image.open(default_detected_image_path)
        st.image(default_detected_image_path, caption = "Detected Image", use_container_width = True)
      else:
        if st.sidebar.button("Delete Objects"):
          result = model.predict(uploaded_image, conf=confidence_value)
          boxes = result[0].boxes
          result_plotted = result[0].plot()[:,:,::-1]
          st.image(result_plotted, caption="Detected Image", use_container_width=True)
          try:
            with st.expander("Detection Results"):
              for box in boxes:
                st.write(box.data)
          except Exception as e:
            st.error("Detection Error")
            st.error(e)
    except Exception as e:
      st.error(e)
elif source_radio == VIDEO:
    source_video = st.sidebar.selectbox(
        "Choose a Video...", VIDEOS_DICT.keys()
    )
    with open(VIDEOS_DICT.get(source_video), 'rb') as video_file:
        video_bytes = video_file.read()
        if video_bytes:
            st.video(video_bytes)
        if st.sidebar.button("Detect Video Objects"):
            try:
                video_cap = cv2.VideoCapture(
                    str(VIDEOS_DICT.get(source_video))
                )
                st_frame = st.empty()
                while (video_cap.isOpened()):
                    success, image = video_cap.read()
                    if success:
                        image = cv2.resize(image, (720, int(720 * (9/16))))
                        #Predict the objects in the image using YOLO11
                        result = model.predict(image, conf = confidence_value)
                        #Plot the detected objects on the video frame
                        result_plotted = result[0].plot()
                        st_frame.image(result_plotted, caption = "Detected Video",
                                       channels = "BGR",
                                       use_container_width=True)
                    else:
                        video_cap.release()
                        break
            except Exception as e:
                st.sidebar.error("Error Loading Video"+str(e))