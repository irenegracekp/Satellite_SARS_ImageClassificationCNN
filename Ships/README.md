##Multiple classification of Ships images

##### The images are in the train folder and a CSV file contains the corresponding labels. 
##### This model contains a ShipDataLoader which takes the CSV file and Images and makes a dataset used for training.

Train folder has 'Cargo', 'Military', 'Carrier', 'cruise', 'Tankers' images which are taken as the different classes.

###Different sections or modules
- Library Imports
- Dataset formation (ShipDataLoader)
- Resnet50, pretrained model initialisation
- Fullyconnected layer definition
- Training 
- Plotting the learning curves

###Specific functions used
- SubsetRandomSampler from torch.utils.data.sampler
- torch.utils.data.DataLoader
- train_test_split from sklearn.model_selection to split training and validation sets  
- models.resnet50(pretrained=True) from torchvision
- torch.optim.lr_scheduler.CosineAnnealingLR

