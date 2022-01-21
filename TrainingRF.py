import cv2
import os 
import numpy as np

dataPath = 'C:/Users/gabri/OneDrive/Escritorio/YAIR/Face_RecognizerOpenCV/data'
listaPersonas =  os.listdir(dataPath)
print ('Lista de Personas: ', listaPersonas)

Etiquetas = []
facesData = []
Etiqueta = 0

for nameDir in listaPersonas:
    personPath = dataPath + '/' + nameDir
    print ('leyendo las imagenes...')

    for fileName in os.listdir(personPath):
        print('Rostros: ',nameDir + '/' + fileName)
        Etiquetas.append(Etiqueta)
        facesData.append(cv2.imread(personPath + '/'+ fileName,0))
        #imagen = cv2.imread(personPath+ '/'+ filename,0)
        #cv2.imshow('imagen',imagen)
        #cv2.waitKey(0)
    Etiqueta = Etiqueta +1

print ('etiquetas= ', Etiquetas)
print ('Numero de etiquetas 0 :', np.count_nonzero(np.array(Etiquetas)==0))
print ('Numero de etiquetas 1 :', np.count_nonzero(np.array(Etiquetas)==1))



