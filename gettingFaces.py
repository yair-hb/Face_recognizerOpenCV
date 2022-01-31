#se debe cambiar al ubicacion de la carpeta donde se encuentran  los programas
#para poder crear la carpeta DATA donde se almacenaran
import cv2
import os 
import imutils

#se crea la carpeta data que almacenara las capturas de los rostros 
folder = 'data'
if not os.path.exists(folder):
    print ('Carpeta creada: ', folder)
    os.makedirs(folder)

namePersona = 'Persona 1' #se debe cambiar el nombre de la persona a la que se le hara la captura de rostro
dataPath = 'C:/Users/gabri/OneDrive/Escritorio/YAIR/Face_RecognizerOpenCV/data'
personPath = dataPath + '/' + namePersona

if not os.path.exists(personPath):
    print ('Carpeta creada: ', personPath)
    os.makedirs(personPath)

captura = cv2.VideoCapture(0, cv2.CAP_DSHOW)
#captura = cv2.VideoCapture('nombre del archivo.mp4')

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_Frontalface_default.xml')
contador = 0

while True:
    ret,frame = captura.read()
    if ret == False:break
    frame = imutils.resize(frame, width=640)
    frameGris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frameAux = frame.copy()

    faces = faceClassif.detectMultiScale(frameGris,1.3,5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y),(x+y,y+h),(0,255,0),2)
        rostro = frameAux [y:y+h,x:x+w]
        rostro = cv2.resize(rostro,(150,150), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(personPath + '/rostro_{}.jpg'.format(contador),rostro)
        contador = contador +1
    cv2.imshow('Capturando rostro', frame)
    k = cv2.waitKey(1)
    if k == 27 or contador >=400:
        break
captura.release()
cv2.destroyAllWindows()
