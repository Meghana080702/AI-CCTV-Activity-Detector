# app.py
import streamlit as st
import cv2
import numpy as np
from datetime import datetime
import pandas as pd
import os
from PIL import Image
import io
import matplotlib.pyplot as plt

# Paths
CFG = "yolov3-tiny.cfg"
WEIGHTS = "yolov3-tiny.weights"
NAMES = "coco.names"
LOG_CSV = "alerts/activity_log.csv"
os.makedirs("alerts", exist_ok=True)

st.set_page_config(layout="wide", page_title="AI CCTV Activity Detector")

st.title("AI CCTV Activity Detector — Demo")
st.markdown(
    "Use **Take a snapshot** (webcam) or **Upload Image**. The app runs YOLOv3-tiny (OpenCV DNN) "
    "and shows annotated frame + logs. Dashboard below shows detection statistics."
)

# --- Load model (robust check) ---
@st.cache_resource(show_spinner=False)
def load_yolo(cfg_path, weights_path, names_path):
    if not os.path.exists(cfg_path) or not os.path.exists(weights_path) or not os.path.exists(names_path):
        return None, None
    net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    with open(names_path, "r") as f:
        classes = [c.strip() for c in f.readlines()]
    layer_names = net.getLayerNames()
    out_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    return net, (classes, out_layers)

net, yolo_meta = load_yolo(CFG, WEIGHTS, NAMES)
if net is None:
    st.error("YOLO model files not found. Make sure yolov3-tiny.cfg, yolov3-tiny.weights and coco.names are in the repo.")
    st.stop()

classes, output_layers = yolo_meta

# Utility: run detection on an OpenCV BGR image
def run_yolo_on_image(bgr_img, conf_thres=0.35, nms_thres=0.4):
    h, w = bgr_img.shape[:2]
    blob = cv2.dnn.blobFromImage(bgr_img, 1/255.0, (320, 320), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            cid = int(np.argmax(scores))
            conf = float(scores[cid])
            if conf > conf_thres:
                cx = int(detection[0]*w)
                cy = int(detection[1]*h)
                bw = int(detection[2]*w)
                bh = int(detection[3]*h)
                x = int(cx - bw/2)
                y = int(cy - bh/2)
                boxes.append([x, y, bw, bh])
                confidences.append(conf)
                class_ids.append(cid)
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_thres, nms_thres)
    return boxes, confidences, class_ids, idxs

# Logging helper
def append_log(obj_name, conf):
    os.makedirs("alerts", exist_ok=True)
    row = {"Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
           "Object": obj_name, "Confidence": f"{conf:.2f}"}
    df = pd.DataFrame([row])
    if not os.path.exists(LOG_CSV):
        df.to_csv(LOG_CSV, index=False)
    else:
        df.to_csv(LOG_CSV, mode="a", header=False, index=False)

# Layout: two columns: left = input+display, right = dashboard
left, right = st.columns((2,1))

with left:
    st.subheader("Camera / Input")
    col_cam, col_upload = st.columns([1,1])
    with col_cam:
        cam_file = st.camera_input("Take a snapshot using your webcam")
    with col_upload:
        uploaded = st.file_uploader("Or upload an image (jpg/png)", type=["jpg","jpeg","png","mp4","mov"])

    input_image = None
    is_video = False

    if cam_file is not None:
        img_bytes = cam_file.getvalue()
        pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        input_image = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    elif uploaded is not None:
        if uploaded.type.startswith("image"):
            file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
            input_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        else:
            st.info("Video uploaded. This demo will process only the first frame of the video as a preview.")
            file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
            vid = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR) if False else None
            # For simplicity, we try to extract a frame with OpenCV VideoCapture from the bytes
            tmp_file = "videos/tmp_upload.mp4"
            os.makedirs("videos", exist_ok=True)
            with open(tmp_file, "wb") as f:
                f.write(file_bytes)
            cap = cv2.VideoCapture(tmp_file)
            ret, frame = cap.read()
            cap.release()
            if ret:
                input_image = frame
            else:
                st.warning("Couldn't decode a frame from the uploaded video.")

    if input_image is None:
        st.info("Take a snapshot or upload an image to run detection.")
    else:
        # Run detection
        boxes, confidences, class_ids, idxs = run_yolo_on_image(input_image, conf_thres=0.35)
        # Draw boxes
        out_img = input_image.copy()
        if isinstance(idxs, (list, tuple, np.ndarray)) and len(idxs) > 0:
            flatten_idxs = idxs.flatten() if hasattr(idxs, "flatten") else idxs
        else:
            flatten_idxs = []
        for i in flatten_idxs:
            x,y,w,h = boxes[i]
            cid = class_ids[i]
            conf = confidences[i]
            label = classes[cid]
            color = tuple(map(int, np.random.randint(0,255,3)))
            cv2.rectangle(out_img, (x,y), (x+w, y+h), color, 2)
            cv2.putText(out_img, f"{label} {conf*100:.0f}%", (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Log detection
            append_log(label, conf*100)

        # Show annotated image
        st.image(cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)

with right:
    st.subheader("Dashboard (Live)")
    if os.path.exists(LOG_CSV):
        df = pd.read_csv(LOG_CSV)
        # normalize headers
        df.columns = df.columns.str.strip()
        if 'Timestamp' in df.columns:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
            df['Hour'] = df['Timestamp'].dt.hour
        else:
            st.warning("Log file exists but header missing. Running quick fix.")
            st.write(df.head())
        st.markdown("**Recent detections (last 10)**")
        st.dataframe(df.tail(10).sort_values("Timestamp", ascending=False))

        st.markdown("**Counts per object**")
        counts = df['Object'].value_counts()
        st.bar_chart(counts)

        st.markdown("**Detections by hour**")
        by_hour = df.groupby(df['Timestamp'].dt.hour).size()
        fig, ax = plt.subplots()
        by_hour.plot(kind="bar", ax=ax)
        ax.set_xlabel("Hour")
        ax.set_ylabel("Detections")
        st.pyplot(fig)
    else:
        st.info("No detections logged yet. Run detection to create alerts/activity_log.csv")

st.markdown("---")
st.caption("Note: This demo uses snapshots (camera input or uploaded images). For continuous streaming see streamlit-webrtc or a WebRTC-based deployment.")
