import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os
import time
from collections import deque
import Quartz

# --- Media Key Constants ---
NX_KEYTYPE_PLAY  = 16
NX_KEYTYPE_NEXT  = 18
NX_KEYTYPE_FAST  = 19

def send_media_key(key_type):
    """Send a media key event (press + release) via Quartz HID."""
    def post_event(down):
        # Media keys are 'NSEvent' subtype 8
        # The key flags/data are packed into the data1 field
        flags = 0xa00 if down else 0xb00
        data1 = (key_type << 16) | flags
        
        ev = Quartz.NSEvent.otherEventWithType_location_modifierFlags_timestamp_windowNumber_context_subtype_data1_data2_(
            14, # NSSystemDefined
            (0, 0), # location
            0xa00 if down else 0, # flags
            0, # timestamp
            0, # window
            0, # context
            8, # subtype for media keys
            data1,
            -1 # data2
        )
        
        # Post the CGEvent version of the NSEvent
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev.CGEvent())

    post_event(True)
    time.sleep(0.01)
    post_event(False)
# --- Constants & Config ---
MODEL_PATH = "hand_landmarker.task"
SWIPE_THRESHOLD = 0.15
HISTORY_SIZE = 10
POSE_CONFIRM_FRAMES = 3   # frames a pose must be held before triggering
ACTION_COOLDOWN = 1.5     # seconds between any media actions

WRIST = 0
THUMB_TIP, THUMB_IP = 4, 3
FINGER_TIPS = [8, 12, 16, 20]
FINGER_PIPS = [6, 10, 14, 18]

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4), (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12), (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20), (5,9),(9,13),(13,17)
]

class HandTracker:
    def __init__(self):
        self.history = deque(maxlen=HISTORY_SIZE)
        self.last_swipe_time = 0
        self.swipe_cooldown = 0.8
        self.swipe_count = 0

    def update_and_detect_swipe(self, wrist_x):
        now = time.time()
        self.history.append((wrist_x, now))
        if now - self.last_swipe_time < self.swipe_cooldown:
            return None
        if len(self.history) == HISTORY_SIZE:
            start_x, _ = self.history[0]
            dx = wrist_x - start_x
            if dx > SWIPE_THRESHOLD:
                self.swipe_count += 1
                self.last_swipe_time = now
                self.history.clear()
                return "RIGHT"
            elif dx < -SWIPE_THRESHOLD:
                self.swipe_count += 1
                self.last_swipe_time = now
                self.history.clear()
                return "LEFT"
        return None

def get_hand_stats(hand, handedness):
    up_count = 0
    if handedness == "Right":
        if hand[THUMB_TIP].x > hand[THUMB_IP].x: up_count += 1
    else:
        if hand[THUMB_TIP].x < hand[THUMB_IP].x: up_count += 1
    for tip, pip in zip(FINGER_TIPS, FINGER_PIPS):
        if hand[tip].y < hand[pip].y: up_count += 1
    if up_count == 0: return "FIST"
    if up_count >= 4: return "OPEN PALM"
    return "STANCE"

# --- Pose debounce state ---
pose_buffer = {"Left": [], "Right": []}
last_action_time = 0
last_triggered_pose = None  # track last pose to avoid re-triggering same pose

def debounced_pose_action(handedness, pose):
    """Trigger a media action only if the same pose is held for POSE_CONFIRM_FRAMES."""
    global last_action_time, last_triggered_pose
    buf = pose_buffer[handedness]
    buf.append(pose)
    if len(buf) > POSE_CONFIRM_FRAMES:
        buf.pop(0)

    now = time.time()
    if (len(buf) == POSE_CONFIRM_FRAMES
            and all(p == pose for p in buf)
            and now - last_action_time > ACTION_COOLDOWN):

        if pose == "OPEN PALM" and last_triggered_pose != "OPEN PALM":
            send_media_key(NX_KEYTYPE_PLAY)
            last_action_time = now
            last_triggered_pose = "OPEN PALM"
            return "▶ PLAY"
        elif pose == "FIST" and last_triggered_pose != "FIST":
            send_media_key(NX_KEYTYPE_PLAY)  # play/pause is a toggle on macOS
            last_action_time = now
            last_triggered_pose = "FIST"
            return "⏸ PAUSE"

    if pose == "STANCE":
        last_triggered_pose = None  # reset when hand goes neutral
    return None

# --- Setup MediaPipe ---
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        MODEL_PATH
    )

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.7,
    running_mode=vision.RunningMode.VIDEO
)

landmarker = vision.HandLandmarker.create_from_options(options)
trackers = {"Left": HandTracker(), "Right": HandTracker()}

cap = cv2.VideoCapture(0)
last_feedback = ""
feedback_until = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    ts = int(time.time() * 1000)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(mp_image, ts)

    current_status = {"Left": "None", "Right": "None"}

    if result.hand_landmarks:
        for i, landmarks in enumerate(result.hand_landmarks):
            handedness = result.handedness[i][0].display_name
            pose = get_hand_stats(landmarks, handedness)
            current_status[handedness] = pose

            # --- Swipe detection (only when palm open) ---
            swipe_dir = None
            if pose == "OPEN PALM":
                swipe_dir = trackers[handedness].update_and_detect_swipe(landmarks[WRIST].x)

            if swipe_dir == "RIGHT":
                now = time.time()
                if now - last_action_time > ACTION_COOLDOWN:
                    send_media_key(NX_KEYTYPE_NEXT)
                    last_action_time = now
                    last_feedback = "⏭ SKIP"
                    feedback_until = now + 1.5

            # --- Pose-based actions ---
            action = debounced_pose_action(handedness, pose)
            if action:
                last_feedback = action
                feedback_until = time.time() + 1.5

            # --- Draw Skeleton ---
            points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
            for start, end in HAND_CONNECTIONS:
                cv2.line(frame, points[start], points[end], (200, 200, 200), 2)
            for cx, cy in points:
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
            cv2.putText(frame, pose, (points[0][0]-30, points[0][1]+30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # --- TOP LEFT: Swipe Counter ---
    total_swipes = trackers["Left"].swipe_count + trackers["Right"].swipe_count
    cv2.rectangle(frame, (10, 10), (250, 60), (0, 0, 0), -1)
    cv2.putText(frame, f"Total Swipes: {total_swipes}", (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # --- TOP RIGHT: Hand Status ---
    cv2.rectangle(frame, (w-260, 10), (w-10, 100), (0, 0, 0), -1)
    cv2.putText(frame, f"L: {current_status['Left']}", (w-240, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"R: {current_status['Right']}", (w-240, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # --- CENTER: Action Feedback ---
    if time.time() < feedback_until:
        text_size = cv2.getTextSize(last_feedback, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3)[0]
        tx = (w - text_size[0]) // 2
        cv2.putText(frame, last_feedback, (tx, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 3)

    cv2.imshow("Hand Control Hub", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

landmarker.close()
cap.release()
cv2.destroyAllWindows()