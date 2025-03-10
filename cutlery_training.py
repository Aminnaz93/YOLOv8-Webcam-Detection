import cv2
import os
import time
import random
import shutil
import subprocess
from ultralytics import YOLO

# ✅ Skapa dataset-mappen om den inte finns
DATASET_DIR = "bestick_dataset"
IMAGE_DIR = os.path.join(DATASET_DIR, "images")
LABEL_DIR = os.path.join(DATASET_DIR, "labels")
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(LABEL_DIR, exist_ok=True)

# ✅ Skapa bestick_dataset.yaml automatiskt
yaml_file_path = os.path.join(DATASET_DIR, "bestick_dataset.yaml")
yaml_content = f"""train: {IMAGE_DIR}/train
val: {IMAGE_DIR}/val

nc: 9  # Antal klasser

names: ["fork", "knife", "spoon", "sas_fork", "sas_knife", "sas_spoon", "emirates_fork", "emirates_knife", "emirates_spoon"]
"""

if not os.path.exists(yaml_file_path):
    with open(yaml_file_path, "w") as f:
        f.write(yaml_content)
    print(f"✅ {yaml_file_path} skapades automatiskt!")

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

# ✅ Starta kameran
cap = cv2.VideoCapture(0)
model = YOLO("yolov8n.pt")

selected_airline = None  # Lagrar valt flygbolag (SAS, Emirates eller None)
training_started = False  # Flagga för att stoppa kameran vid träning

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
                label = BESTICK_CLASSES[cls]  # Hämta besticknamn

                # ✅ Om flygbolag är valt, ändra klassnummer
                if selected_airline == "sas":
                    new_cls = cls + 58
                    label = BESTICK_CLASSES[new_cls]
                elif selected_airline == "emirates":
                    new_cls = cls + 158
                    label = BESTICK_CLASSES[new_cls]
                else:
                    new_cls = cls

                # ✅ Spara bild
                timestamp = str(int(time.time()))
                img_filename = f"{IMAGE_DIR}/{label}/{label}_{timestamp}.jpg"
                label_filename = f"{LABEL_DIR}/{label}/{label}_{timestamp}.txt"
                cv2.imwrite(img_filename, frame)

                # ✅ Spara annoteringsfil
                x_center = (box.xyxy[0][0] + box.xyxy[0][2]) / (2 * frame.shape[1])
                y_center = (box.xyxy[0][1] + box.xyxy[0][3]) / (2 * frame.shape[0])
                width = (box.xyxy[0][2] - box.xyxy[0][0]) / frame.shape[1]
                height = (box.xyxy[0][3] - box.xyxy[0][1]) / frame.shape[0]

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

    # ✅ Tangenter för val
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
    elif key == ord('t'):
        print("🚀 Stänger av kameran och startar dataset-splitting och träning...")
        training_started = True
        break  # Avbryter loopen för att stänga av kameran

cap.release()
cv2.destroyAllWindows()  # ✅ Stäng av kameran
print(f"✅ training_started = {training_started}")

if training_started:
    # ✅ 1. SPLITTA DATASETET
    print("📂 Delar upp datasetet i träning och validering...")

    train_path = os.path.join(IMAGE_DIR, "train")
    val_path = os.path.join(IMAGE_DIR, "val")
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)

    train_labels_path = os.path.join(LABEL_DIR, "train")
    val_labels_path = os.path.join(LABEL_DIR, "val")
    os.makedirs(train_labels_path, exist_ok=True)
    os.makedirs(val_labels_path, exist_ok=True)

    # ✅ Lista bilder och blanda
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith((".jpg", ".png"))]
    random.shuffle(image_files)

    # ✅ 80% träning, 20% validering
    split_index = int(0.8 * len(image_files))
    train_files, val_files = image_files[:split_index], image_files[split_index:]

    # ✅ Flytta filer
    for file in train_files + val_files:
        src_img = os.path.join(IMAGE_DIR, file)
        dest_folder = "train" if file in train_files else "val"
        shutil.move(src_img, os.path.join(IMAGE_DIR, dest_folder, file))

        label_file = file.replace(".jpg", ".txt").replace(".png", ".txt")
        if os.path.exists(os.path.join(LABEL_DIR, label_file)):
            shutil.move(os.path.join(LABEL_DIR, label_file), os.path.join(LABEL_DIR, dest_folder, label_file))

    print("✅ Datasetet har delats in i 80% träning och 20% validering!")

    # ✅ 2. STARTA YOLO-TRÄNINGEN
    print("🚀 Startar YOLO-träning...")
    subprocess.run([
        "yolo", "train",
        "model=yolov8n.pt",
        f"data={yaml_file_path}",
        "epochs=50",
        "imgsz=640",
        "device=cpu",
        "--resume"
    ])
    print("🎉 Träningen är klar!")