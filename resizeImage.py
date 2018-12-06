import numpy as np
import cv2
import os

PATH_EYES = "base/lara2/"
DST = "base/laramin2/"


img_paths = os.listdir(PATH_EYES)
contador = 1
for ip in img_paths:
	img = cv2.imread(PATH_EYES+ip)
	ip_png = str(DST)+str(ip)
	reflect =cv2.resize(img,(224, 224), interpolation = cv2.INTER_CUBIC)
	cv2.imwrite(ip_png,reflect)
	contador+=1
	print("Imagem: ",contador);