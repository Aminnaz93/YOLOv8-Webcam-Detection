import cv2
import os
import time
import subprocess
from ultralytics import YOLO

# ✅ Skapa bestick_dataset.yaml automatiskt om den saknas
yaml_content = """train: bestick_dataset/images/train  # Sökväg till träningsbilder
val: bestick_dataset/images/val      # Sökväg till valideringsbilder

nc: 9  # Antal klasser (3 vanliga + 3 SAS + 3 Emirates)

names: ["fork", "knife", "spoon", "sas_fork", "sas_knife", "sas_spoon", "emirates_fork", "emirates_knife", "emirates_spoon"]
"""

yaml_file_path = "bestick_dataset.yaml"
if not os.path.exists(yaml_file_path):
    with open(yaml_file_path, "w") as f:
        f.write(yaml_content)
    print(f"✅ {yaml_file_path} skapades automatiskt!")

# ✅ Ladda YOLO-modellen
model = YOLO("yolov8n.pt")

# ✅ Skapa mappar för dataset (bilder + annoteringar)
DATASET_DIR = "bestick_dataset"
IMAGE_DIR = f"{DATASET_DIR}/images"
LABEL_DIR = f"{DATASET_DIR}/labels"
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(LABEL_DIR, exist_ok=True)

# ✅ Klassnummer för vanliga bestick, SAS och Emirates
BESTICK_CLASSES = {
    42: "fork", 43: "knife", 44: "spoon",
    100: "sas_fork", 101: "sas_knife", 102: "sas_spoon",
    200: "emirates_fork", 201: "emirates_knife", 202: "emirates_spoon"
}

# ✅ Skapa mappar för varje besticktyp
for category in BESTICK_CLASSES.values():
    os.makedirs(f"{IMAGE_DIR}/{category}", exist_ok=True)
    os.makedirs(f"{LABEL_DIR}/{category}", exist_ok=True)

cap = cv2.VideoCapture(0)
selected_airline = None  # Standard: Ingen flygbolagsklassning

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

            if cls in [42, 43, 44]:  # Endast bestick
                bestick_identifierat = True
                label = BESTICK_CLASSES[cls]

                # ✅ Ändra klass beroende på flygbolag
                if selected_airline == "sas":
                    new_cls = cls + 58  # Ex: fork (42) → sas_fork (100)
                    label = BESTICK_CLASSES[new_cls]
                elif selected_airline == "emirates":
                    new_cls = cls + 158  # Ex: fork (42) → emirates_fork (200)
                    label = BESTICK_CLASSES[new_cls]
                else:
                    new_cls = cls  # Behåll vanlig bestickklass

                # ✅ Skapa filnamn
                timestamp = str(int(time.time()))
                img_filename = f"{IMAGE_DIR}/{label}/{label}_{timestamp}.jpg"
                label_filename = f"{LABEL_DIR}/{label}/{label}_{timestamp}.txt"

                # ✅ Spara bilden
                cv2.imwrite(img_filename, frame)

                # ✅ Normalisera bounding box
                x_center = (box.xyxy[0][0] + box.xyxy[0][2]) / (2 * frame.shape[1])
                y_center = (box.xyxy[0][1] + box.xyxy[0][3]) / (2 * frame.shape[0])
                width = (box.xyxy[0][2] - box.xyxy[0][0]) / frame.shape[1]
                height = (box.xyxy[0][3] - box.xyxy[0][1]) / frame.shape[0]

                # ✅ Spara annoteringsfil
                with open(label_filename, "w") as f:
                    f.write(f"{new_cls} {x_center} {y_center} {width} {height}\n")

                print(f"✅ Sparade {img_filename} och {label_filename} med klass {new_cls}")

                # ✅ Rita en rektangel runt besticket
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # ✅ Visa kameraflödet
    cv2.imshow("YOLO - Bestickidentifiering", frame)

    # 📌 Tangenter för funktioner
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        selected_airline = "sas"
        print("🔵 SAS-bestick valda!")
    elif key == ord('e'):
        selected_airline = "emirates"
        print("🔴 Emirates-bestick valda!")
    elif key == ord('n'):
        selected_airline = None
        print("⚪ Vanliga bestick valda!")
    elif key == ord('t'):  # Tryck 't' för att starta träningen
        print("🚀 Startar träning...")
        cap.release()
        cv2.destroyAllWindows()
        subprocess.run(["python3", "train_model.py"])
        break

cap.release()
cv2.destroyAllWindows()