# import cv2
# import numpy as np
# from tensorflow.keras.models import load_model

# # =========================
# # Load Trained Model
# # =========================
# model = load_model("DriveGuard_EfficientNet10.h5")
# class_names = ['c0','c1','c2','c3','c4','c5','c6','c7','c8','c9']  # Adjust if needed
# CONFIDENCE_THRESHOLD = 0.6

# # =========================
# # Initialize Webcam
# # =========================
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     img = cv2.resize(frame, (224,224))
#     img = img / 255.0
#     img = np.expand_dims(img, axis=0)

#     preds = model.predict(img)
#     class_idx = np.argmax(preds[0])
#     confidence = preds[0][class_idx]

#     if confidence >= CONFIDENCE_THRESHOLD:
#         label = class_names[class_idx]
#         cv2.putText(frame, f"{label}: {confidence*100:.2f}%", (10,30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

#     cv2.imshow("DriveGuard Real-Time Detection", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


import cv2
import numpy as np
from tensorflow.keras.models import load_model

# =========================
# Load Trained Model
# =========================
model = load_model("DriveGuard_EfficientNet10.h5")
class_names = [
    'safe driving',
    'texting - right',
    'talking on the phone - right',
    'texting - left',
    'talking on the phone - left',
    'operating the radio',
    'drinking',
    'reaching behind',
    'hair and makeup',
    'talking to passenger'
]
CONFIDENCE_THRESHOLD = 0.6

# =========================
# Initialize Webcam
# =========================
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, (224,224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img, verbose=0)
    class_idx = np.argmax(preds[0])
    confidence = preds[0][class_idx]

    if confidence >= CONFIDENCE_THRESHOLD:
        label = class_names[class_idx]
        # ✅ Color coding
        if class_idx == 0:
            color = (0, 255, 0)  # Green for safe
            warning_text = "Safe driving"
        else:
            color = (0, 0, 255)  # Red for unsafe
            warning_text = f"⚠ Distracted! ({label})"

        # Overlay text
        cv2.putText(frame, f"{label} ({confidence*100:.2f}%)", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        if class_idx != 0:
            cv2.putText(frame, warning_text, (10,70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("DriveGuard Real-Time Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

