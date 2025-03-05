import cv2
import os
import time
from ultralytics import YOLO

# Ladda befintlig YOLO-modell
model = YOLO("yolov8n.pt")

# Skapa mappar för dataset (bilder + annoteringar)
DATASET_DIR = "bestick_dataset"
IMAGE_DIR = f"{DATASET_DIR}/images"
LABEL_DIR = f"{DATASET_DIR}/labels"
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(LABEL_DIR, exist_ok=True)

# Klassnummer för vanliga bestick, SAS och Emirates
BESTICK_CLASSES = {
    42: "fork", 43: "knife", 44: "spoon",
    100: "sas_fork", 101: "sas_knife", 102: "sas_spoon",
    200: "emirates_fork", 201: "emirates_knife", 202: "emirates_spoon"
}

# Skapa mappar för varje besticktyp
for category in BESTICK_CLASSES.values():
    os.makedirs(f"{IMAGE_DIR}/{category}", exist_ok=True)
    os.makedirs(f"{LABEL_DIR}/{category}", exist_ok=True)

cap = cv2.VideoCapture(0)

selected_airline = None  # Lagrar valt flygbolag (SAS, Emirates eller None)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    bestick_identifierat = False  

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = box.conf[0].item()

            if cls in [42, 43, 44]:  # Om det är ett bestick
                bestick_identifierat = True
                label = BESTICK_CLASSES[cls]  # Hämta besticknamn (fork, knife, spoon)

                # Om du har valt flygbolag, ge besticket ett nytt klassnummer
                if selected_airline == "sas":
                    new_cls = cls + 58  # Exempel: fork (42) → sas_fork (100)
                    label = BESTICK_CLASSES[new_cls]
                elif selected_airline == "emirates":
                    new_cls = cls + 158  # Exempel: fork (42) → emirates_fork (200)
                    label = BESTICK_CLASSES[new_cls]
                else:
                    new_cls = cls  # Vanliga bestick behåller sin klass

                # Skapa filnamn
                timestamp = str(int(time.time()))
                img_filename = f"{IMAGE_DIR}/{label}/{label}_{timestamp}.jpg"
                label_filename = f"{LABEL_DIR}/{label}/{label}_{timestamp}.txt"

                # Spara bilden
                cv2.imwrite(img_filename, frame)

                # Normalisera bounding box-koordinater för YOLO-format
                x_center = (box.xyxy[0][0] + box.xyxy[0][2]) / (2 * frame.shape[1])
                y_center = (box.xyxy[0][1] + box.xyxy[0][3]) / (2 * frame.shape[0])
                width = (box.xyxy[0][2] - box.xyxy[0][0]) / frame.shape[1]
                height = (box.xyxy[0][3] - box.xyxy[0][1]) / frame.shape[0]

                # Spara annoteringsfil med korrekt klassnummer
                with open(label_filename, "w") as f:
                    f.write(f"{new_cls} {x_center} {y_center} {width} {height}\n")

                print(f"✅ Sparade {img_filename} och {label_filename} med klass {new_cls}")

                # Rita en rektangel runt besticket
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Visa kameraflödet
    cv2.imshow("YOLO - Bestickidentifiering", frame)

    # Tangenter för att välja flygbolag
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):  # Tryck 's' för SAS
        selected_airline = "sas"
        print("🔵 SAS-bestick valda!")
    elif key == ord('e'):  # Tryck 'e' för Emirates
        selected_airline = "emirates"
        print("🔴 Emirates-bestick valda!")
    elif key == ord('n'):  # Tryck 'n' för att återgå till vanliga bestick
        selected_airline = None
        print("⚪ Vanliga bestick valda!")

cap.release()
cv2.destroyAllWindows()

