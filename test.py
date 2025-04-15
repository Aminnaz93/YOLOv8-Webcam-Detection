#activate python virtual enviroment in command line - source yolovenv/bin/activate
#Deactivate from python virtual enviroment , type in - deactivate

from ultralytics import YOLO

model = YOLO("yolov8n.pt")

result = model("/home/pi/Desktop/cutlerysorter/test_image.jpg" , save=True)

print(result)