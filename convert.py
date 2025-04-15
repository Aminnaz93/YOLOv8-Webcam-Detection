#convert from pytorch format to ncnn format

from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.export(format="ncnn")

