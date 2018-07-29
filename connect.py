# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 18:50:34 2018

@author: trabz
"""
from time import sleep as sl
import os
import numpy as np
import cv2
from keras.models import model_from_json
from keras.models import load_model
from keras import backend as K
K.set_image_dim_ordering('th')
PATH = os.getcwd()+'\model\d2model.hdf5'
# Define data path

img_rows=128
img_cols=128
num_channel=3

loaded_model=load_model(PATH)
#test_image1 = cv2.imread('C:/Users/trabz/Desktop/4.jpg')
lis2=['100TL','10TL','20TL','50TL','5tl'];
cam = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
while True:
    ret, test_image1 =cam.read()
    test_image=cv2.resize(test_image1,(128,128))
    #test_image=cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    test_image = np.array(test_image)
    test_image = test_image.astype('float32')
    test_image /= 255
    print (test_image.shape)     
    test_image=np.rollaxis(test_image,2,0)
    #test_image= np.expand_dims(test_image, axis=0)
    test_image= np.expand_dims(test_image, axis=0)
    print (test_image.shape)
    conf=loaded_model.predict(test_image)
    print(conf)
   
    if conf[0,int(loaded_model.predict_classes(test_image))]>0.75:
     print((loaded_model.predict(test_image)))
     Id2=lis2[int(loaded_model.predict_classes(test_image))]
     cv2.putText(test_image1, Id2, (300,300), font, 3, (255, 0,2),5)
    cv2.imshow('im',test_image1) 
    if cv2.waitKey(100) ==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()



#test_image=cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
