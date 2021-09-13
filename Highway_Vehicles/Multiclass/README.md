##Multiple classification of images of vehicles on Highway

#### This program takes datafrom a json fiw with the labels and makes the dataset from a common image folder of all images.

Train folder has 'Four-Wheeler', 'Two-Wheeler', 'High Load', 'Medium Load', 'Three-Wheeler', 'Bus', 'Non-Vehicle Image' as the 7 different classes.

###Different sections or modules
- Library Imports
- Making Dataset from Json fil and common Image Folder
- CNN Model definition
- Training 
- Plotting the learning curves

###Specific functions used
- VehicleDataset(self defined) function for making dataset.
- train_test_split from sklearn.model_selection to split training and validation sets  

