import os
import random
import shutil

# Stig till din dataset-mapp
dataset_path = "/home/pi/Desktop/CutlerySorter/dataset"  # Ändra till rätt sökväg
images_path = os.path.join(dataset_path, "images")
labels_path = os.path.join(dataset_path, "labels")

# Skapa mappar för YOLO-strukturen
os.makedirs(os.path.join(images_path, "train"), exist_ok=True)
os.makedirs(os.path.join(images_path, "val"), exist_ok=True)
os.makedirs(os.path.join(labels_path, "train"), exist_ok=True)
os.makedirs(os.path.join(labels_path, "val"), exist_ok=True)

# Lista alla bildfiler (förutsätter att de slutar på .jpg eller .png)
image_files = [f for f in os.listdir(images_path) if f.endswith((".jpg", ".png"))]

# Blanda listan slumpmässigt
random.shuffle(image_files)

# Räkna ut hur många som ska vara i träning (80%) och validering (20%)
split_index = int(0.8 * len(image_files))
train_files = image_files[:split_index]
val_files = image_files[split_index:]

# Flytta bilder och annoteringsfiler
for file in train_files:
    shutil.move(os.path.join(images_path, file), os.path.join(images_path, "train", file))
    label_file = file.replace(".jpg", ".txt").replace(".png", ".txt")  # Hitta motsvarande YOLO-txt
    if os.path.exists(os.path.join(labels_path, label_file)):
        shutil.move(os.path.join(labels_path, label_file), os.path.join(labels_path, "train", label_file))

for file in val_files:
    shutil.move(os.path.join(images_path, file), os.path.join(images_path, "val", file))
    label_file = file.replace(".jpg", ".txt").replace(".png", ".txt")  # Hitta motsvarande YOLO-txt
    if os.path.exists(os.path.join(labels_path, label_file)):
        shutil.move(os.path.join(labels_path, label_file), os.path.join(labels_path, "val", label_file))

print("✅ Datasetet har delats in i 80% träning och 20% validering!")