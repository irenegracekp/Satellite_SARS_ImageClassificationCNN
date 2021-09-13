"""for multi classification of ships images"""
import copy
import os
import matplotlib.pyplot as plt
import torch.optim as optim
import torch
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
from torchvision import models
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from PIL import Image
os.getcwd()

import json
# Opening JSON file 
with open('./inputs/sample.json', 'r') as openfile: 
  
    # Reading from json file 
    json_object = json.load(openfile) 

batchSize = json_object['batchSize']
numEpoch = json_object['numEpoch']
learningRate = json_object['learningRate']

trainData = pd.read_csv('inputs/train.csv')

ship = {1: 'Cargo', 2: 'Military', 3: 'Carrier', 4: 'Cruise', 5: 'Tankers'}

trainData['category'].value_counts()

label = 'Cargo', 'Military', 'Carrier', 'cruise', 'Tankers'
plt.figure(figsize=(8, 8))
plt.pie(trainData.groupby('category').size(), labels=label, autopct='%1.1f%%', shadow=True, startangle=90)
plt.show()


def pil_loader(path):
	"""opening image from folder"""
	with open(path, 'rb') as imageFile:
		img = Image.open(imageFile)
		return img.convert('RGB')


class ShipDataLoader(Dataset):
	"""making the images data with the images"""
	def __init__(self, csvfolder, process='train', transform=transforms.Compose([transforms.Resize(size=(224, 224)), transforms.ToTensor()]), imgFolder='inputs/train/', labelsDict={}, ylabel=list(trainData.category)):

		self.process = process
		self.imgFolder = imgFolder
		self.csvfolder = csvfolder
		self.ylabel = ylabel
		self.fileList = pd.read_csv(self.csvfolder)['image'].tolist()
		self.transform = transform
		self.labelsDict = labelsDict

		if self.process == 'train':
			self.labels = [labelsDict[i] for i in self.fileList]
		else:
			self.labels = [0 for i in range(len(self.fileList))]

	def __len__(self):
		return len(self.fileList)

	def __getitem__(self, idx):
		fileName = self.fileList[idx]
		imageData = pil_loader(self.imgFolder + "/" + fileName)

		if self.transform:
			imageData = self.transform(imageData)

		if self.process == 'train':
			label1 = self.ylabel[idx]
		else:
			label1 = fileName

		return imageData, label1


def imshow(img, title):
	"""display function"""
	npimg = img.numpy()
	plt.figure(figsize=(15, 15))
	plt.axis("off")

	plt.imshow(np.transpose(npimg, (1, 2, 0)))
	plt.title(title, fontsize=15)
	plt.show()


def show_batch_images(dataloader):
	"""display batch of images function"""
	images, labels = next(iter(dataloader))
	img = torchvision.utils.make_grid(images)
	imshow(img, "classes: " + str([str(x.item()) + " " + ship[x.item()] for x in labels]))


trainingBatchsize = batchSize
mapImgClassDict = {k:v for k, v in zip(trainData.image, trainData.category)}
fullData = ShipDataLoader('inputs/train.csv', process="train", imgFolder="inputs/train", labelsDict=mapImgClassDict)

trainfulLoader = torch.utils.data.DataLoader(fullData, batch_size=trainingBatchsize, shuffle=True)

show_batch_images(trainfulLoader)


# In[9]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

transformOri = transforms.Compose([transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])


tr, val = train_test_split(trainData.category, stratify=trainData.category, test_size=0.15, random_state=10)


trainingBatchsize = 16
numWorkers = 8

trainSampler = SubsetRandomSampler(list(tr.index))
validSampler = SubsetRandomSampler(list(val.index))

traindataset = ShipDataLoader('inputs/train.csv', "train", transformOri, 'inputs/train', mapImgClassDict)
trainLoader = torch.utils.data.DataLoader(traindataset, batch_size=trainingBatchsize, sampler=trainSampler, num_workers=0)

valdataset = ShipDataLoader('inputs/train.csv', "train", transformOri, 'inputs/train', mapImgClassDict)
validLoader = torch.utils.data.DataLoader(valdataset, batch_size=trainingBatchsize, sampler=validSampler, num_workers=0)

show_batch_images(trainLoader)
show_batch_images(validLoader)


model = models.resnet50(pretrained=True)

print(model)

fc = nn.Sequential(nn.Linear(model.fc.in_features, 720), nn.ReLU(), nn.Dropout(0.5), nn.Linear(720, 256), nn.ReLU(), nn.Dropout(0.4), nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 5), nn.Softmax(dim=1))

model.fc = fc

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


model = model.to(device)
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), learningRate, weight_decay=1e-3)

expLrScheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.15)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(trainLoader), eta_min=1e-6)

trainSizes = len(list(tr.index))
validSizes = len(list(val.index))


# In[ ]:

trainAcc = []
validAcc = []
trainLosses = []
validLosses = []
trainAccuracy = []
validAccuracy = []
bestModel = copy.deepcopy(model)
bestAcc = 0.0
for epoch in range(1, numEpoch + 1):
	# keep-track-of-training-and-validation-loss
	trainLoss = 0.0
	validLoss = 0.0
	trainAcc = 0.0
	validAcc = 0.0
	correct = 0
	correct1 = 0
	# training-the-model
	scheduler.step()
	model.train()
	for data, target in trainLoader:
		data = data.to(device)
		target = target.to(device) - 1
		optimizer.zero_grad()
		torch.set_grad_enabled(True)
		# forward-pass: compute-predicted-outputs-by-passing-inputs-to-the-model
		output = model(data)
		# calculate-the-batch-loss
		loss = criterion(output, target)
		# backward-pass: compute-gradient-of-the-loss-wrt-model-parameters
		loss.backward()
		optimizer.step()
		# update-training-loss

		trainLoss += loss.item() * data.size(0)
		_, predicted = torch.max(output.data, 1)
		correct += torch.sum(predicted == target.data)

	model.eval()
	for data, target in validLoader:
		data = data.to(device)
		target = target.to(device) - 1
		torch.set_grad_enabled(False)
		output = model(data)
		loss = criterion(output, target)
		# update-average-validation-loss
		validLoss += loss.item() * data.size(0)
		_, predicted = torch.max(output.data, 1)
		correct1 += torch.sum(predicted == target.data)
	# calculate-average-losses
	trainLoss = trainLoss / trainSizes
	validLoss = validLoss / validSizes
	trainAcc = correct.double() / trainSizes
	validAcc = correct1.double() / validSizes
	trainLosses.append(trainLoss)
	validLosses.append(validLoss)
	trainAccuracy.append(100 * trainAcc)
	validAccuracy.append(100 * validAcc)

	print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}  {:.6f}   {:.6f} '.format(epoch, trainLoss, validLoss, trainAccuracy[-1], validAccuracy[-1]))

	if validAcc > bestAcc:
		bestAcc = validAcc
		bestModel = copy.deepcopy(model)


# In[10]:


plt.plot(trainAccuracy, label='Training accuracy')
plt.plot(validAccuracy, label='Validation accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(frameon=False)


# In[11]:


plt.plot(trainLosses, label='Training Loss')
plt.plot(validLosses, label='Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(frameon=False)
