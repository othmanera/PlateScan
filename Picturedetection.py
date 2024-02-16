import cv2
import numpy as np
from ultralytics import YOLO
from sort.sort import *
from util import *


# tracker
vehicule_tracker = Sort()


#models
yolo_Model= YOLO('yolov8n.pt') #default cocoset YOLO model
license_plate_detector = YOLO('best.pt') #custom model for license plate detection

# load picture
frame = cv2.imread('test.jpg')


vehicules = [2,3,5,7]
#read frames
frame_nbr = -1
ret = True






# vehicules detections

detections = yolo_Model(frame)
detections_ = []
for detection in detections.boxes.data.tolist():
        x1,y1,x2,y2,score,class_id= detection
        if int(class_id) in vehicules:
            detections_.append([x1,y1,x2,y2,score])


# detect license plate
license_plates = license_plate_detector(frame)[0]
for license_plate in license_plates.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = license_plate


# crop license plate
LP_crop = frame[int(y1):int(y2), int(x1):int(x2),:]

#process license plate
license_plate_crop_gray = cv2.cvtColor(LP_crop, cv2.COLOR_BGR2GRAY)
_, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

cv2.imshow('original_crop',LP_crop)
cv2.imshow('threshold', LP_crop)