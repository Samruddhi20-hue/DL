import cv2
import tensorflow as tf
import numpy as np

# Load model
model = tf.keras.models.load_model("you_vs_notyou_cnn.h5")

IMG_SIZE = 100  # must match training size
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)  # 0 = default webcam
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (IMG_SIZE, IMG_SIZE)).astype(np.float32)/255.0
        inp = np.expand_dims(face_resized, axis=(0, -1))

        prob = model.predict(inp, verbose=0)[0][0]
        label = "YOU" if prob < 0.5 else "NOT YOU"
        color = (0,255,0) if label == "YOU" else (0,0,255)

        cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
        cv2.putText(frame, f"{label} ({prob:.2f})", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Webcam Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
