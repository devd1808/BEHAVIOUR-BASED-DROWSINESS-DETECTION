import cv2
from mtcnn.mtcnn import MTCNN
from pygame import mixer
import os
from keras.models import load_model
import numpy as np
import pygame
mixer.init()
pygame.mixer.music.load('ambu.mpeg')
cap=cv2.VideoCapture(0)
model = load_model('cnncat2.h5')
path = os.getcwd()
detector=MTCNN()
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count=0
thicc=2
rpred=[99]
lpred=[99]
p=-1
l=[1 for _ in range(100)]
countc,counto=0,0
while True:
    ret,img=cap.read()
    height1, width1 = img.shape[:2]
    faces = detector.detect_faces(img)
    for face in faces:
        x, y, width, height = face['box']
        x2, y2 = x + width, y + height
        cv2.rectangle(img, (x, y), (x2, y2), (255, 0, 0), 4)
        break
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
    reye = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_righteye_2splits.xml')
    leye = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_lefteye_2splits.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    right_eye = reye.detectMultiScale(gray)
    left_eye = leye.detectMultiScale(gray)
    if(faces!=[]):
     o=1
     for (x, y, w, h) in right_eye:
         r_eye = img[y:y + h, x:x + w]
         count=count+1
         r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
         r_eye = cv2.resize(r_eye, (24, 24))
         r_eye = r_eye / 255
         r_eye = r_eye.reshape(24, 24, -1)
         r_eye = np.expand_dims(r_eye, axis=0)
         predict_x = model.predict(r_eye)
         rpred = np.argmax(predict_x, axis=1)
         if (rpred[0] == 1):
             lbl = 'Open'
         if (rpred[0] == 0):
             lbl = 'Closed'
         break
     for (x, y, w, h) in left_eye:
        l_eye = img[y:y + h, x:x + w]
        count=count+1
        l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
        l_eye = cv2.resize(l_eye, (24, 24))
        l_eye = l_eye / 255
        l_eye = l_eye.reshape(24, 24, -1)
        l_eye = np.expand_dims(l_eye, axis=0)
        predict_x = model.predict(l_eye)
        lpred = np.argmax(predict_x, axis=1)
        if (lpred[0] == 1):
            lbl = 'Open'
        if (lpred[0] == 0):
            lbl = 'Closed'
        break
     if (rpred[0] == 0 and lpred[0] == 0):
            o=0
            cv2.putText(img, "Closed", (10, height - 20), font, 1, (255, 0, 0), 1, cv2.LINE_AA)
            countc+=2
            counto=0
            # if(rpred[0]==1 or lpred[0]==1):
     else:
            cv2.putText(img, "Open", (10, height - 20), font, 1, (255, 0, 0), 1, cv2.LINE_AA)
            countc=0
            counto+=3

     for i in range(countc):
      l.remove(l[0])
      l+=[o]
     for i in range(counto):
         l.remove(l[0])
         l+=[o]
     p=sum(l)/len(l)
     cv2.putText(img, 'Percentage:' + str(round((p)*100,2))+" %.", (100, height - 20), font, 1, (255, 0, 0), 1, cv2.LINE_AA)
     if(p<0.65):
             try:
                 if (pygame.mixer.music.get_busy() == False):
                     pygame.mixer.music.play(0)
             except:  # isplaying = False
                 pass
             if (thicc < 16):
                 thicc = thicc + 2
             else:
                 thicc = thicc - 2
                 if (thicc < 2):
                     thicc = 2
             cv2.rectangle(img, (0, 0), (width1, height1), (0, 0, 255), thicc)
    else:
        cv2.putText(img, "FACE NOT DETECTED!!", (80, 250), font, 2, (0,255, 0), 3, cv2.LINE_AA)
    cv2.imshow('Video', img)
    k = cv2.waitKey(40) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()