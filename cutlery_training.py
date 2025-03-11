import cv2
import os
import time
import random
import shutil
import subprocess
from ultralytics import YOLO

# Skapa dataset-mappen om den inte finns
DATASET_DIR = "/Users/aminnazari/Desktop/Python/Kamera/bestick_dataset"
IMAGE_DIR = os.path.join(DATASET_DIR, "images")
LABEL_DIR = os.path.join(DATASET_DIR, "labels")

for subdir in ["train", "val"]:
    os.makedirs(os.path.join(IMAGE_DIR, subdir), exist_ok=True)
    os.makedirs(os.path.join(LABEL_DIR, subdir), exist_ok=True)

# Skapa bestick_dataset.yaml automatiskt
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

# Klassnummer för vanliga bestick, SAS och Emirates
BESTICK_CLASSES = {
    42: "fork", 43: "knife", 44: "spoon",
    100: "sas_fork", 101: "sas_knife", 102: "sas_spoon",
    200: "emirates_fork", 201: "emirates_knife", 202: "emirates_spoon"
}

# Mappa klassnummer till YOLO-klasser
CLASS_MAPPING = {
    42: 0,  43: 1,  44: 2,  # COCO-klasser
    100: 3, 101: 4, 102: 5,  # SAS-klasser
    200: 6, 201: 7, 202: 8   # Emirates-klasser
}

# Starta kameran och ladda YOLO-modellen
cap = cv2.VideoCapture(0)
model = YOLO("yolov8n.pt")

selected_airline = None  # Flygbolagsval
training_started = False  # Flagga för att stoppa kameran vid träning

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = box.conf[0].item()

            if cls in [42, 43, 44]:  # Om det är ett bestick
                label = BESTICK_CLASSES[cls]

                # Anpassa klassnummer beroende på flygbolag
                if selected_airline == "sas":
                    new_cls = cls + 58
                    label = BESTICK_CLASSES[new_cls]
                elif selected_airline == "emirates":
                    new_cls = cls + 158
                    label = BESTICK_CLASSES[new_cls]
                else:
                    new_cls = cls

                # Mappa till YOLO-klasser
                if new_cls in CLASS_MAPPING:
                    new_cls = CLASS_MAPPING[new_cls]
                else:
                    continue  # Skippa om klassen inte finns i vår mappning

                # Spara bild och label
                timestamp = str(int(time.time()))
                img_filename = f"{label}_{timestamp}.jpg"
                label_filename = f"{label}_{timestamp}.txt"

                img_path = os.path.join(IMAGE_DIR, "train", img_filename)
                label_path = os.path.join(LABEL_DIR, "train", label_filename)

                cv2.imwrite(img_path, frame)

                x_center = (box.xyxy[0][0] + box.xyxy[0][2]) / (2 * frame.shape[1])
                y_center = (box.xyxy[0][1] + box.xyxy[0][3]) / (2 * frame.shape[0])
                width = (box.xyxy[0][2] - box.xyxy[0][0]) / frame.shape[1]
                height = (box.xyxy[0][3] - box.xyxy[0][1]) / frame.shape[0]

                with open(label_path, "w") as f:
                    f.write(f"{new_cls} {x_center} {y_center} {width} {height}\n")

                print(f"Sparade {img_path} och {label_path} med klass {new_cls}")

                # Rita en grön rektangel runt besticket
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Visa kameraflödet
    cv2.imshow("YOLO - Bestickidentifiering", frame)

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
        break  

cap.release()
cv2.destroyAllWindows()

if training_started:
    # SPLITTA DATASETET I TRÄNING OCH VALIDERING
    print("📂 Delar upp datasetet i träning och validering...")

    image_files = [f for f in os.listdir(os.path.join(IMAGE_DIR, "train")) if f.endswith((".jpg", ".png"))]
    random.shuffle(image_files)

    split_index = int(len(image_files) * 0.8)
    train_files = image_files[:split_index]
    val_files = image_files[split_index:]

    def move_files(files, dest_folder):
        for file in files:
            src_img = os.path.join(IMAGE_DIR, "train", file)
            dest_img = os.path.join(IMAGE_DIR, dest_folder, file)
            shutil.move(src_img, dest_img)

            label_file = file.replace(".jpg", ".txt").replace(".png", ".txt")
            src_label = os.path.join(LABEL_DIR, "train", label_file)
            dest_label = os.path.join(LABEL_DIR, dest_folder, label_file)

            if os.path.exists(src_label):
                shutil.move(src_label, dest_label)

    move_files(val_files, "val")

    print("Datasetet har delats in i 80% träning och 20% validering!")

    # ✅ STARTA YOLO-TRÄNING
    print("Startar YOLO-träning...")
    subprocess.run([
        "yolo", "train",
        "model=yolov8n.pt",
        f"data={yaml_file_path}",
        "epochs=50",
        "imgsz=640",
        "device=cpu",
        "pretrained=True"
    ])
    print("🎉 Träningen är klar!")