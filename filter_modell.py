#filtrerat bort allt förutom bestick

from ultralytics import YOLO
import cv2

# Ladda din YOLOv8n-modell
model = YOLO("yolov8n.pt")

# Lista över klasser vi vill behålla (COCO ID: 42=fork, 43=knife, 44=spoon)
BESTICK_CLASSES = [42, 43, 44]

# Starta kameran
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Kör YOLO på bilden
    results = model(frame)

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])  # Klassens index
            conf = box.conf[0].item()  # Sannolikhet
            label = result.names[cls]  # Klassnamn

            # ✅ Endast behåll bestick
            if cls in BESTICK_CLASSES:
                print(f"✅ Identifierat: {label} med {conf:.2f} sannolikhet")
                cv2.putText(frame, f"{label} {conf:.2f}", (int(box.xyxy[0][0]), int(box.xyxy[0][1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("YOLOv8 - Bestick Identifiering", frame)

    # Avbryt med 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()