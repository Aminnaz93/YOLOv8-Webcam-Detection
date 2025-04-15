#activate python virtual enviroment in command line - source yolovenv/bin/activate
#Deactivate from python virtual enviroment , type in - deactivate

from ultralytics import YOLO

#model = YOLO("yolov8n.pt")
model = YOLO("/Users/aminnazari/Desktop/Python/Kamera/yolov8n_ncnn_model")
             
result = model("/Users/aminnazari/Desktop/Python/Kamera/test_image.jpg" , save=True)

print(result)