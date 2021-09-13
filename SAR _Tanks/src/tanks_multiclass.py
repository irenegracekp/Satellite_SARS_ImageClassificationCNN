"""for multi classification of Tanks from SAR images"""
import copy
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import json
# Opening JSON file 
with open('./inputs/sample.json', 'r') as openfile: 
  
    # Reading from json file 
    json_object = json.load(openfile) 

batchSize = json_object['batchSize']
numEpoch = json_object['numEpoch']
learningRate = json_object['learningRate']

transformOri = transforms.Compose([transforms.Resize((64, 64)), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

trainDataset = datasets.ImageFolder(root='./inputs/train', transform=transformOri)


# In[2]:


trainData, validData = train_test_split(trainDataset, train_size=0.6)
trainLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=batchSize, shuffle=True, num_workers=0)
validLoader = torch.utils.data.DataLoader(dataset=validData, batch_size=batchSize, shuffle=False, num_workers=0)


# In[7]:


class CNN(nn.Module):
	"""making the CNN model"""

	def __init__(self):
		super(CNN, self).__init__()
		self.cnn1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1)
		self.batchnorm1 = nn.BatchNorm2d(8)
		self.relu = nn.ReLU()
		self.maxpool1 = nn.MaxPool2d(kernel_size=2)
		self.cnn2 = nn.Conv2d(in_channels=8, out_channels=32, kernel_size=5, stride=1, padding=2)
		self.batchnorm2 = nn.BatchNorm2d(32)
		self.maxpool2 = nn.MaxPool2d(kernel_size=2)
		self.cnn3 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3, stride=1, padding=1)
		self.batchnorm3 = nn.BatchNorm2d(128)
		self.relu = nn.ReLU()
		self.maxpool3 = nn.MaxPool2d(kernel_size=2)
		self.fc1 = nn.Linear(in_features=8192, out_features=4000)
		self.fc2 = nn.Linear(in_features=4000, out_features=2000)
		self.droput = nn.Dropout(p=0.5)
		self.fc3 = nn.Linear(in_features=2000, out_features=500)
		self.fc4 = nn.Linear(in_features=500, out_features=50)
		self.droput = nn.Dropout(p=0.5)
		self.fc5 = nn.Linear(in_features=50, out_features=2)
		self.fc6 = nn.Sequential(nn.Linear(8192, 2000), nn.ReLU(), nn.Dropout(0.5), nn.Linear(2000, 500), nn.ReLU(), nn.Dropout(0.4), nn.Linear(500, 125), nn.ReLU(), nn.Dropout(0.3), nn.Linear(125, 30), nn.ReLU(), nn.Dropout(0.2), nn.Linear(30, 7), nn.Dropout(0.1), nn.Linear(7, 4), nn.Softmax(dim=1))

	def forward(self, x):
		out = self.cnn1(x)
		out = self.batchnorm1(out)
		out = self.relu(out)
		out = self.maxpool1(out)
		out = self.cnn2(out)
		out = self.batchnorm2(out)
		out = self.relu(out)
		out = self.maxpool2(out)
		out = self.cnn3(out)
		out = self.batchnorm3(out)
		out = self.relu(out)
		out = self.maxpool3(out)
		out = out.view(-1, 8192)
		out = self.fc6(out)
		return out


# In[8]:


model = CNN()
# pylint: disable=E1101
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# pylint: enable=E1101
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), learningRate, weight_decay=1e-3)
# Decay LR by a factor of 0.15 every 7 epochs
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(trainLoader), eta_min=1e-6)
trainSizes = len(trainLoader.sampler)
validSizes = len(validLoader.sampler)


# In[9]:


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
		target = target.to(device)
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
		target = target.to(device)
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
