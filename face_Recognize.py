#-----------ejecutar gettinFaces.py (captura las fotos de las personas a detectar)
#-----------ejecutar Training.py (crea el modelo xml para el tipo de metodo a usar)
#-----------ejecutar face_recognizer.py 
#----------------------se habilita el condicionante con los parametros de deteccion correctos 

import cv2
import os

from cv2 import INTER_CUBIC

#***********************cambia la direccion por la ubicacion de la carpeta DATA en tu equipo
dataPath = 'C:/Users/gabri/OneDrive/Escritorio/YAIR/Face_RecognizerOpenCV/data'
imagePaths = os.listdir(dataPath)
print ('imagePaths = ', imagePaths)
print ('Rostros a encontrar:', imagePaths)

#face_recognizer = cv2.face.EigenFaceRecognizer_create()
#face_recognizer = cv2.face.FisherFaceRecognizer_create()
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

#********************se lee el modelo cerado para cada uno de los metodos
#face_recognizer.read('modeloEigenFace.xml')
#face_recognizer.read('modeloFisherFace.xml')
face_recognizer.read('modeloLBPHFace.xml')

#********************se hace la lectura ya sea de camara web o de algun video con los rostros 
captura = cv2.VideoCapture(0, cv2.CAP_DSHOW)
#captura = cv2.VideoCapture('video.mp4')

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

while True:
    ret, frame = captura.read()
    if ret == False:
        break
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frameAux = gris.copy()

    faces = faceClassif.detectMultiScale(gris,1.3,5)

    for (x,y,w,h) in faces:
        rostro = frameAux[y:y+h, x:x+w]
        rostro = cv2.resize(rostro, (150,150), interpolation=INTER_CUBIC)
        resultado = face_recognizer.predict(rostro)

        cv2.putText(frame , '{}'.format(resultado),(x,y-5),1,1.3,(255,255,0),1,cv2.LINE_AA)

        #**************Elije cual es el metodo a usar para el reconocimiento de rostro
        '''
        #EigenFaces
        if resultado [1] <5700:
            cv2.putText(frame, '{}'.format(imagePaths[resultado[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
            cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        else:
            cv2.putText(frame, 'Desconocido', (x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
            cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)

        #FisherFaces 
        if resultado [1] <500:
            cv2.putText(frame, '{}'.format(imagePaths[resultado[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
            cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        else:
            cv2.putText(frame,'Desconocido',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
            cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
        '''
        #LBPH Metod
        if resultado [1] < 70:
            cv2.putText(frame, '{}'.format(imagePaths[resultado[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
            cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        else:
            cv2.putText(frame, 'Desconocido', (x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
            cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
        
    cv2.imshow('Reconociendo rostros',frame)
    k =  cv2.waitKey(1)
    if k == 27:
        break
captura.release()
cv2.destroyAllWindows()
