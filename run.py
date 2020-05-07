from torch.autograd import Variable
import cv2           
import matplotlib.pyplot as plt
from matplotlib import cm
import torchvision.models as models
import torch.nn as nn
import torch
import os
import numpy as np
from torchvision import datasets
from torchvision import transforms
import torch.optim as optim
from glob import glob
from trainm import Net
from PIL import Image, ImageEnhance
import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
def detect_faces(cascade, test_image, scaleFactor = 1.1):
	image_copy = test_image.copy()
	faces_rect = cascade.detectMultiScale(image_copy[:,:,2], scaleFactor=scaleFactor, minNeighbors=5 ,	minSize = (30,30))
	flag = 0
	for (x, y, w, h) in faces_rect:
		#print("Hello wait untill Machine detects Mask")
		crop_img = image_copy[y:y+h, x:x+w]	
		flag = 1
		flag , val = Classify(crop_img)

		if flag == 0:
			crop_img = cv2.putText(image_copy, f'Thank you!{max(val*100)}', (x,y-10),  cv2.FONT_HERSHEY_SIMPLEX , 0.5 ,(0,255, 0) ,2 , cv2.LINE_AA) 
			cv2.rectangle(image_copy, (x, y), (x+w, y+h), (0, 255, 0), 1)

		elif flag == 1:
			crop_img = cv2.putText(image_copy, f'Wear Mask{max(val*100)}', (x,y-10),  cv2.FONT_HERSHEY_SIMPLEX , 0.5 ,(255, 0,0) ,2 , cv2.LINE_AA) 
			cv2.rectangle(image_copy, (x, y), (x+w, y+h), (255, 0, 0), 5)

		else:
			crop_img = cv2.putText(image_copy, f'Wear Mask  Properly{max(val*100)}', (x,y-10),  cv2.FONT_HERSHEY_SIMPLEX , 0.5 ,(0, 0 , 255) ,2 , cv2.LINE_AA) 
			cv2.rectangle(image_copy, (x, y), (x+w, y+h), (0,0, 255), 5)

	return image_copy

def Classify(image):
	data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
	image = Image.open('asd.jpg')
	size = (224, 224)
	image = ImageOps.fit(image, size, Image.ANTIALIAS)
	image_array = np.asarray(image)
	normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
	data[0] = normalized_image_array
	prediction = model.predict(data)
	print(softmax(prediction))

	if prediction[0][1] > prediction[0][0] and prediction[0][1] > prediction[0][2]:
		return 0 , max(softmax(prediction))
	elif prediction[0][0] > prediction[0][1] and prediction[0][0] > prediction[0][2]:
		return 1 , max(softmax(prediction))
	else:
		return 2 , max(softmax(prediction))

if __name__ == '__main__':
	cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	#videofile = 'Clip1.wmv'
	out = cv2.VideoWriter('outpy_intern_mirasys.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 20, (256,256))
	cap = cv2.VideoCapture(0)
	#This Net model is Made from Scratch but due to complexity of detection (As we need higher accuracy) we used pretrained Models
	model_transfer = Net()
	#Three Class Classification : wear mask : wear mask properly : thank you for wearing mask
	model = tensorflow.keras.models.load_model('Trained_mask.h5')
	while(True):
		ret, frame = cap.read()
		cv2.imwrite('asd.jpg' , frame)
		gray = detect_faces(cascade,frame)
		cv2.imshow('frame',gray)
		out.write(frame)
		cv2.waitKey(500)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	cap.release()
	cv2.destroyAllWindows()
np.set_printoptions(suppress=True)
