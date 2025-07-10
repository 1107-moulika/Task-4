import cv2
import numpy as np
from sort import Sort
from ultralytics import YOLO

# Load pre-trained YOLOv8 model
model = YOLO("yolov8n.pt")

# Initialize video capture
cap = cv2.VideoCapture(0)  # Use 0 for webcam or replace with 'video.mp4'

# Initialize SORT tracker
tracker = Sort()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv8 object detection
    results = model(frame)[0]
    detections = []

    for result in results.boxes:
        x1, y1, x2, y2 = result.xyxy[0].cpu().numpy()
        conf = result.conf[0].cpu().numpy()
        cls = int(result.cls[0].cpu().numpy())
        detections.append([x1, y1, x2, y2, conf])

    # Convert detections to NumPy array
    if len(detections) > 0:
        detections = np.array(detections)
    else:
        detections = np.empty((0, 5))

    # Update tracker
    tracks = tracker.update(detections)

    # Draw bounding boxes and labels
    for track in tracks:
        x1, y1, x2, y2, track_id = track.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Object Detection and Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
