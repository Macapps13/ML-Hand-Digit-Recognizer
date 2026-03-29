import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.hands()
mp_draw = mp.solutions.drawing_utils

tip_ids = [4, 8, 12, 16, 20]

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLORBGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = hand_landmarks.landmark
            fingers = []

            if landmarks[tip_ids[0]].x < landmarks[tip_ids[0] - 1].x:
                fingers.append(1)
            else:
                fingers.append(0)
            
            for i in range(1, 5):
                if landmarks[tip_ids[i]].y < landmarks[tip_ids[i] - 2].y:
                    fingers.append(1)
                else:
                    fingers.append(0)
            
            total_fingers = fingers.count(1)

            cv2.rectangle(img, (20, 20), (170, 120), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, str(total_fingers), (45, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 5)
        
    cv2.imshow("Hand Gesture Recognition", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
            
