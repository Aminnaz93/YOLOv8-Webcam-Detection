import os
import random
import shutil

# 📌 Sätt rätt dataset-mapp
dataset_path = "/Users/aminnazari/Desktop/Python/Kamera/bestick_dataset"
images_path = os.path.join(dataset_path, "images/train")  # Ändrat till "train"
labels_path = os.path.join(dataset_path, "labels/train")  # Ändrat till "train"

# 📂 Skapa mappar för YOLO-strukturen
os.makedirs(os.path.join(dataset_path, "images/train"), exist_ok=True)
os.makedirs(os.path.join(dataset_path, "images/val"), exist_ok=True)
os.makedirs(os.path.join(dataset_path, "labels/train"), exist_ok=True)
os.makedirs(os.path.join(dataset_path, "labels/val"), exist_ok=True)

# 🔍 Lista alla bilder i "train"
image_files = [f for f in os.listdir(images_path) if f.endswith((".jpg", ".png"))]
print(f"🔍 Hittade {len(image_files)} bilder i {images_path}")

# ❌ Om inga bilder hittas, avbryt
if not image_files:
    print("❌ Inga bilder att dela upp!")
    exit()

# 🔀 Blanda listan slumpmässigt
random.shuffle(image_files)

# 📊 Räkna ut hur många som ska vara i träning (80%) och validering (20%)
split_index = int(0.8 * len(image_files))
train_files = image_files[:split_index]
val_files = image_files[split_index:]

# 🚀 Flytta bilder och annoteringsfiler
for file in image_files:
    src_img = os.path.join(images_path, file)
    dest_folder = "train" if file in train_files else "val"
    dest_img = os.path.join(dataset_path, "images", dest_folder, file)

    print(f"📂 Flyttar {file} till {dest_folder}...")
    shutil.move(src_img, dest_img)

    # 🏷️ Flytta motsvarande label-fil
    label_file = file.replace(".jpg", ".txt").replace(".png", ".txt")
    src_label = os.path.join(labels_path, label_file)
    dest_label = os.path.join(dataset_path, "labels", dest_folder, label_file)

    if os.path.exists(src_label):
        print(f"📂 Flyttar label {label_file} till {dest_folder}...")
        shutil.move(src_label, dest_label)
    else:
        print(f"⚠️ Label saknas för {file}")

print("✅ Datasetet har delats in i 80% träning och 20% validering!")