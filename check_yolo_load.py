import cv2

try:
    net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
    print("✅ YOLO model loaded successfully!")
except Exception as e:
    print("❌ Error loading YOLO:", e)
