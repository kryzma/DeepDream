import torch
from torchvision import models


class GoogLeNet(torch.nn.Module):

	def __init__(self):
		super().__init__()

		googlenet = models.googlenet(pretrained=True)

		self.used_layers = ['inception3b', 'inception4c', 'inception4d', 'inception4e']

		self.layers = dict()

		self.layers['conv1'] = googlenet.conv1
		self.layers['maxpool1'] = googlenet.maxpool1
		self.layers['conv2'] = googlenet.conv2
		self.layers['conv3'] = googlenet.conv3
		self.layers['maxpool2'] = googlenet.maxpool2
		self.layers['inception3a'] = googlenet.inception3a
		self.layers['inception3b'] = googlenet.inception3b
		self.layers['maxpool3'] = googlenet.maxpool3
		self.layers['inception4a'] = googlenet.inception4a
		self.layers['inception4b'] = googlenet.inception4b
		self.layers['inception4c'] = googlenet.inception4c
		self.layers['inception4d'] = googlenet.inception4d
		self.layers['inception4e'] = googlenet.inception4e

		for param in self.parameters():
			param.requires_grad = False


	def forward(self, x):
		output = []
		for layer_name, layer in self.layers.items():
			x = layer(x)
			if layer_name in self.used_layers:
				output.append(x)

		return output

