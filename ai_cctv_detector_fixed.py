import cv2
import numpy as np

# Load YOLOv3-tiny model
net = cv2.dnn.readNetFromDarknet("yolov3-tiny.cfg", "yolov3-tiny.weights")

# Use CPU backend
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Load class names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Get output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Test image
img = cv2.imread("test.jpg")
if img is None:
    print("❌ Image not found. Place 'test.jpg' in the same folder.")
    exit()

height, width, _ = img.shape
blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

class_ids, confidences, boxes = [], [], []

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.3:
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = int(detection[0] * width - w / 2)
            y = int(detection[1] * height - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)

font = cv2.FONT_HERSHEY_SIMPLEX
for i in indexes.flatten():
    x, y, w, h = boxes[i]
    label = f"{classes[class_ids[i]]} {confidences[i]*100:.1f}%"
    color = (0, 255, 0)
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    cv2.putText(img, label, (x, y - 10), font, 0.5, color, 2)

cv2.imshow("YOLO Detection Test", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
