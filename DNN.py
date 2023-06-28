# model and weight link: https://github.com/opencv/opencv/tree/master/samples/dnn
# inspiration taken from: https://towardsdatascience.com/face-detection-models-which-to-use-and-why-d263e82c302c

import cv2
import numpy as np

model = "models/res10_300x300_ssd_iter_140000.caffemodel"
config = "models/deploy.prototxt.txt"

net = cv2.dnn.readNetFromCaffe(config, model)

cap = cv2.VideoCapture(0)
while True:
    _, img = cap.read()
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,(300, 300), (104.0, 117.0, 123.0))
    net.setInput(blob)
    faces = net.forward()
    #to draw faces on image
    for i in range(faces.shape[2]):
            confidence = faces[0, 0, i, 2]
            if confidence > 0.5:
                box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x1, y1) = box.astype("int")
                cv2.rectangle(img, (x, y), (x1, y1), (0, 0, 255), 2)
    
    cv2.imshow('img', img)
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break


cap.release()