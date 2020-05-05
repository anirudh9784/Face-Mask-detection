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
from train import Net
from PIL import Image
def predict_breed_transfer(img):
	# load the image and return the predicted breed
	img = Image.fromarray(img.astype(np.uint8))
	transform = transforms.Compose([
			transforms.Resize(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225])
			])
	img_tensor = transform(img)
    
	# reshaping to include batch size
	img_tensor = img_tensor.view(1, img_tensor.shape[0], img_tensor.shape[1], img_tensor.shape[2])
	
	prediction = model_transfer(img_tensor)
	
	class_idx = torch.argmax(prediction).item()
	print(prediction , class_idx)
	return class_idx
def detect_faces(cascade, test_image, scaleFactor = 1.1):
	# create a copy of the image to prevent any changes to the original one.
	image_copy = test_image.copy()
	#convert the test image to gray scale as opencv face detector expects gray images
	#gray_image = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)

	# Applying the haar classifier to detect faces
	faces_rect = cascade.detectMultiScale(image_copy[:,:,1], scaleFactor=scaleFactor, minNeighbors=5)
	flag = 0
	for (x, y, w, h) in faces_rect:
		#cv2.rectangle(image_copy, (x, y), (x+w, y+h), (0, 255, 0), 15)
		crop_img = image_copy[y:y+h, x:x+w]
		flag = 1
	if flag:
		if predict_breed_transfer(crop_img):
			cv2.rectangle(image_copy, (x, y), (x+w, y+h), (0, 255, 0), 15)
			print('Mask')
		else:
			cv2.rectangle(image_copy, (x, y), (x+w, y+h), (255,0, 0), 15)
			print('No Mask')
		return image_copy
	else:
		return image_copy

if __name__ == '__main__':
	cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	#videofile = 'Clip1.wmv'
	cap = cv2.VideoCapture('WIN_20200505_23_44_58_Pro.mp4')
	model_transfer = Net()
	model_transfer.load_state_dict(torch.load('model.pt') )
	while(True):
		# Capture frame-by-frame
		ret, frame = cap.read()

		# Our operations on the frame come here
		gray = detect_faces(cascade,frame)
		# Display the resulting frame
		cv2.imshow('frame',gray)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	# When everything done, release the capture
	cap.release()
	cv2.destroyAllWindows()