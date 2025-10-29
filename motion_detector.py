import cv2
import numpy as np

# Open webcam or CCTV feed
cap = cv2.VideoCapture(0)  # 0 means default webcam. Replace with video path or CCTV stream URL

# Initialize background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Apply background subtraction
    fgmask = fgbg.apply(frame)

    # Remove noise and smooth the mask
    kernel = np.ones((5, 5), np.uint8)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.dilate(fgmask, kernel, iterations=2)

    # Find contours (moving objects)
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes for motion
    for contour in contours:
        if cv2.contourArea(contour) > 1000:  # Ignore small movements
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Motion Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2)

    # Show video feed
    cv2.imshow('CCTV Motion Detector', frame)
    cv2.imshow('Foreground Mask', fgmask)

    # Press 'q' to quit
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
