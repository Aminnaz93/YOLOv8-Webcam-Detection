from ultralytics import YOLO
import cv2

# Ladda den tränade modellen
model = YOLO("/Users/aminnazari/Desktop/Python/Kamera/yolov8n.pt")

# Starta webbkameran
cap = cv2.VideoCapture(0)  # 0 = Första kameran

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Kör YOLO-inferens på kamerabilden
    results = model(frame)

    # Visa resultatet
    annotated_frame = results[0].plot()
    cv2.imshow("YOLOv8 Webcam Detection", annotated_frame)

    # Avsluta med 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()