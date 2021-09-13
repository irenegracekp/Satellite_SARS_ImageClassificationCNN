##Multiple classification of SAR images of Tanks

##### This model can be trained to classify any objects that are separated in 4 different directories.
##### The speciality is that the images in different directories are automatically taken as 4 different classes by the program.

Train folder has '2S1', 'BRDM_2', 'SLICY', 'ZSU_23_4' which are taken as the different classes.

###Different sections or modules
- Library Imports
- Images loading
- CNN Model definition
- Training 
- Plotting the learning curves

###Specific functions used
- torchvision.datasets.ImageFolder for loading images from different directories as classes <br />
- train_test_split from sklearn.model_selection to split training and validation sets  

