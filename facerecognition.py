import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path ="a folder for students pictures"
images=[]
classNames = []
myList=os.listdir(path)
#print(myList)

for cl in myList:
    #format the names to strings''
    curImg=cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

#print(classNames)

def findencodings(images):
    encodings=[]
    for img in images:
        img=cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        encode = face_recognition.face_encodings(img)[0]
        encodings.append(encode)
    return encodings

knownencodinglist=findencodings(images)
print('encoding completed')

capture = cv2.VideoCapture(0)

while True:
    success, img =capture.read()
    #resize for easy read 
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    #convert 
    imgS =cv2.cvtColor(imgS, cv2.COLOR_RGB2BGR)

    #get faces from the webcam and get its encodings
    #incase there is more than one face 
    currframefaces =face_recognition.face_locations(imgS)
    currfaceencoding = face_recognition.face_encodings(imgS,currframefaces)

    for encodeFace,faceLoc in zip(currfaceencoding,currframefaces):
        matches =face_recognition.compare_faces(knownencodinglist,encodeFace)
        faceDis = face_recognition.face_distance(knownencodinglist,encodeFace)
        #lowest encodings of the list is our best match 
        #print(faceDis)
        #min is index 0 = first element in the array 
        bestmatch =np.argmin(faceDis)


        if matches[bestmatch]:
            name = classNames[bestmatch].upper()
            print(name)
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            #draw a rectangle
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            #display name 
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)


    cv2.imshow('webcam',img)
    cv2.waitKey(1)





