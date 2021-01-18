import cv2 as cv
import numpy as np

import network as network
import torch
from torchvision import transforms

from PIL import Image

#============== Preparing Images ============#

def preprocess_image(img):
	preprocess = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])
	return preprocess(img)

def deprocess_image(img):
	img *= torch.Tensor([0.229, 0.224, 0.225])
	img += torch.Tensor([0.485, 0.456, 0.406])
	return img

def load_image(img_path, config):
	img = Image.open(img_path)
	img = img.resize((config['height'], config['height']) ,Image.ANTIALIAS)
	return img

def save_image(img, cnt, config):

	path = config['output_dir'] + "img" + str(cnt) + ".jpg" 

	img.save(path)
