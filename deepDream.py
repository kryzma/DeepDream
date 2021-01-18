import torch
from torch.autograd import Variable
from torchvision import models

import os

import utils
import cv2

from PIL import Image, ImageFilter, ImageChops

import numpy as np

def gradient_ascent(img, model, config):

	optimizing_img = Variable(utils.preprocess_image(img).unsqueeze(0), requires_grad=True)

	model.zero_grad()

	for i in range(config["iterations"]):
		out = optimizing_img
		for j in range(config["target_layer"]):
			out = list(model.features.modules())[j+1](out)
		loss = out.norm()
		loss.backward()
		optimizing_img.data = optimizing_img.data + config["learning_rate"] * optimizing_img.grad.data

	optimizing_img = optimizing_img.data.squeeze()
	optimizing_img.transpose_(0, 1)
	optimizing_img.transpose_(1, 2)
	optimizing_img = np.clip(utils.deprocess_image(optimizing_img), 0, 1)
	img = Image.fromarray(np.uint8(optimizing_img*255))
	return img


def deep_dream(config):

	if torch.cuda.is_available():
		device = torch.device("cuda")
	else:
		device = torch.device("cpu")

	model = models.vgg16(pretrained=True).to(device)

	init_image_path = os.path.join(config['init_images_dir'], config['init_image_name'])

	init_img = utils.load_image(init_image_path, config)

	octaves = [init_img]

	# get all needed octaves
	for i in range(config["octaves_cnt"]):

		size = octaves[-1].size[:2]
		size = (int(size[0] // config["octave_scale"]), int(size[1] // config["octave_scale"]) )
		octaves.append(octaves[-1].resize(size, Image.ANTIALIAS))

	octaves = octaves[::-1] # reverse list go from smallest to biggest image

	# apply gradient ascent and merge octaves
	while len(octaves) > 1:

		oct1 = octaves.pop(0)
		oct2 = octaves.pop(0)

		oct1 = gradient_ascent(oct1, model, config)
		oct2 = gradient_ascent(oct2, model, config)

		# join smaller octave to bigger
		oct1 = oct1.resize(oct2.size, Image.ANTIALIAS)

		oct12 = ImageChops.blend(oct1, oct2, config["blend_rate"])
		oct12 = gradient_ascent(oct12, model, config)
		octaves.insert(0, np.array(oct12))

		utils.save_image(oct12, len(octaves), config)




if __name__ == "__main__":

	config = dict()
	config['output_dir'] = "C:/Users/Mantas/Desktop/DeepDream/images/results/"

	config['init_images_dir'] = "C:/Users/Mantas/Desktop/DeepDream/images/contents/"
	config['init_image_name'] = "me.jpg"

	config['height'] = 512

	config['learning_rate'] = 0.09
	config["iterations"] = 10


	config["octaves_cnt"] = 8
	config["octave_scale"] = 1.3
	config["blend_rate"] = 0.5

	config["target_layer"] = 28

	deep_dream(config)
