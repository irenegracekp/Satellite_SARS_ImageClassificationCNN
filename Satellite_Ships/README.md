##Binary classification of Satellite images of Ships

##### This model can be trained to classify any objects that are separated in two different directories.
##### The speciality is that the images in different directories are automatically taken as two different classes by the program.

Train folder has Planes and No_planes images(2 classes used by model)

###Different sections or modules
- Library Imports
- Images loading
- CNN Model definition
- Training 
- Plotting the learning curves

###Specific functions used
- torchvision.datasets.ImageFolder for loading images from different directories as classes <br />
- train_test_split from sklearn.model_selection to split training and validation sets  

