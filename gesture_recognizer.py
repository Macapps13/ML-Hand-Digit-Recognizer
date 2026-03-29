import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os

# Hand connections for drawing skeleton
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),       # thumb
    (0,5),(5,6),(6,7),(7,8),       # index
    (0,9),(9,10),(10,11),(11,12),  # middle
    (0,13),(13,14),(14,15),(15,16),# ring
    (0,17),(17,18),(18,19),(19,20),# pinky
    (5,9),(9,13),(13,17)           # palm
]

model_path = "hand_landmarker.task"
if not os.path.exists(model_path):
    print("Downloading hand landmarker model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        model_path
    )
    print("Done.")

def draw_landmarks(frame, result):
    if not result.hand_landmarks:
        return frame
    h, w, _ = frame.shape
    for hand in result.hand_landmarks:
        points = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
        for start, end in HAND_CONNECTIONS:
            cv2.line(frame, points[start], points[end], (255, 255, 255), 2)
        for cx, cy in points:
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
    return frame

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    running_mode=vision.RunningMode.VIDEO
)

landmarker = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Trying camera index 1...")
    cap = cv2.VideoCapture(1)

print("Press 'q' to quit.")
frame_index = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_index += 1
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
    if timestamp_ms == 0:
        timestamp_ms = frame_index * 33

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    frame = draw_landmarks(frame, result)
    cv2.imshow("Hand Detection", frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

landmarker.close()
cap.release()
cv2.destroyAllWindows()