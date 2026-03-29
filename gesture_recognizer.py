import tensorflow as tf
import cv2
import numpy as np

model = tf.keras.models.load_model('model.h5')

cap = cv2.VideoCapture(0)
cv2.startWindowThread()

while True:
    ret, frame = cap.read()
    if not ret: break

    height, width, _ = frame.shape
    size = 300
    x1, y1 = (width - size) // 2, (height - size) // 2
    x2, y2 = x1 + size, y1 + size

    roi = frame[y1:y2, x1:x2]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    img_resized = cv2.resize(gray, (28, 28))
    img_final = img_resized / 255.0
    img_final = img_final.reshape(1, 28, 28)

    prediction = model.predict(img_final, verbose=0)
    digit = np.argmax(prediction)
    confidence = np.max(prediction)

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = f"Digit: {digit} ({confidence*100:.2f}%)"
    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Gesture Control Input", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 
