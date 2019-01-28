# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 10:27:39 2019

@author: harshul agarwal
"""

import dlib
import cv2
import os
print(os.path.isfile('C:\Users\harshul agarwal\Desktop\i3.jpg'))
cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()

    #img=cv2.resize(img,(1000,700))

    path='C:\Users\harshul agarwal\Desktop\shape_predictor_68_face_landmarks.dat'
    detector=dlib.get_frontal_face_detector()
    predictor=dlib.shape_predictor(path)

    det=detector(img,1)
    print(len(det))
    font = cv2.FONT_HERSHEY_SIMPLEX

    for d in det:
        shape=predictor(img,d)
        for i in range(68):
            x,y=shape.part(i).x,shape.part(i).y
            img1=cv2.putText(img,str(i),(x,y), font, 0.3,(255,0,255),1,cv2.LINE_AA)
            cv2.imshow('a',img1)

    '''dist1=shape.part(64).x-shape.part(48).x
    dist2=shape.part(57).y-shape.part(51).y

    print(dist1)
    if(dist1>95):
        print("happy")
        img1=cv2.putText(img1,"happy",(10,10), font, 0.3,(255,0,255),1,cv2.LINE_AA)
    elif(dist2>40):
        print('surprised')
        img1=cv2.putText(img1,"surprised",(10,10), font, 0.3,(255,0,255),1,cv2.LINE_AA)

    else:
        print('normal')'''
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
#cv2.waitKey()      
