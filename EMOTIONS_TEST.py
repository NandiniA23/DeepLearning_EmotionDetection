#importing a sample of test images.
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Dense,Activation
import cv2

model = keras.models.load_model("Models\EMOTIONS\weights-improvement-16-0.58.hdf5")
video=cv2.VideoCapture(0)
face_cascade=cv2.CascadeClassifier("Cascades/haarcascade_frontalface_default.xml")
while True:
    ret,frame=video.read()
    cv2.imshow("Output",frame)
    faces=face_cascade.detectMultiScale(frame)#x,y,w,h
    for (x,y,w,h) in faces:
       roi=frame[y:y+h,x:x+w].copy()
       cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),3)
       cv2.imshow("Output",frame)
    new=cv2.resize(roi,(48,48))
    norm=new/255
    print(norm.shape)
    prediction=model.predict_classes(norm.reshape(1,48,48,3))
    print(prediction)
    if prediction==[[0]]:
         #cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)
         cv2.putText(frame, 'ANGER', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
         cv2.imshow("Output",frame)
    if prediction==[[1]]:
          #cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)
          cv2.putText(frame, 'FEAR', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
          cv2.imshow("Output",frame)
    if prediction==[[2]]:
          #cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)
          cv2.putText(frame, 'HAPPY', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
          cv2.imshow("Output",frame)
    if prediction==[[3]]:
          #cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)
          cv2.putText(frame, 'SAD', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
          cv2.imshow("Output",frame)
    if prediction==[[4]]:
          #cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)
          cv2.putText(frame, 'SUPRISE', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
          cv2.imshow("Output",frame)
    k=cv2.waitKey(1)
    if k==27:
        break;
    
