import cv2
import os
import imutils

personName = "" #insert down the name of the person
dataPath = "" #insert down the path of the data
personPath = dataPath + "/" + personName

if not os.path.exists(personPath):
    print("Folder Created")
    os.makedirs(personPath)

cap = cv2.VideoCapture('') #path to the video to take the screenshots of the face. 

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
count = 0

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = frame.copy()

    faces = faceClassif.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, h), (0, 255, 0))
        face = auxFrame[y:y + h, x: x + w]
        face = cv2.resize(face, (150, 150), interpolation=cv2.INTER_CUBIC) #determinar tamaño de los frames // determine frame size
        cv2.imwrite(personPath + '/face_{}.jpg'.format(count), face) #conteo de los frames // frame count
        count = count + 1
    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1)
    if k == 27 or count >= 300: #tomar 300 frames // take 300 frames (0-299)
        break

cap.release()
cv2.destroyAllWindows()

#archivo creado por Andres Pachano
#su uso queda permitido con fines educativos mas no comerciales

#script created by Andres Pachano
#the usage of this script is aviable with educative proposes but with no commercial proposes
