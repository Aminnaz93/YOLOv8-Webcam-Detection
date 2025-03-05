import os
import random
import shutil
import subprocess

print("📂 Delar upp datasetet i träning och validering...")

dataset_path = "bestick_dataset"
images_path = os.path.join(dataset_path, "images")
labels_path = os.path.join(dataset_path, "labels")

# ✅ Skapa mappar för YOLO-strukturen
os.makedirs(os.path.join(images_path, "train"), exist_ok=True)
os.makedirs(os.path.join(images_path, "val"), exist_ok=True)
os.makedirs(os.path.join(labels_path, "train"), exist_ok=True)
os.makedirs(os.path.join(labels_path, "val"), exist_ok=True)

# ✅ Lista bilder
image_files = [f for f in os.listdir(images_path) if f.endswith((".jpg", ".png"))]
random.shuffle(image_files)

# ✅ 80% träning, 20% validering
split_index = int(0.8 * len(image_files))
train_files, val_files = image_files[:split_index], image_files[split_index:]

# ✅ Flytta bilder och etiketter
for file in train_files + val_files:
    src_img = os.path.join(images_path, file)
    dest_folder = "train" if file in train_files else "val"
    shutil.move(src_img, os.path.join(images_path, dest_folder, file))

    label_file = file.replace(".jpg", ".txt").replace(".png", ".txt")
    if os.path.exists(os.path.join(labels_path, label_file)):
        shutil.move(os.path.join(labels_path, label_file), os.path.join(labels_path, dest_folder, label_file))

print("✅ Datasetet har delats in i 80% träning och 20% validering!")

# ✅ Starta YOLO-träning
print("🚀 Startar YOLO-träning...")
subprocess.run(["yolo", "train", "model=yolov8n.pt", "data=bestick_dataset.yaml", "epochs=50", "imgsz=640", "resume=True"])
print("🎉 Träningen är klar!")