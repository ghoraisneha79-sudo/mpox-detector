There are 3 folders in the dataset.

1) Original Images: It contains a total number of 228 images, among which 102 belongs to the 'Monkeypox' class and
 the remaining 126 represents the 'Others' class i.e., non-monkeypox (chickenpox and measles) cases.

2) Augmented Images: To aid the classification task, several data augmentation methods such as rotation, translation,
 reflection, shear, hue, saturation, contrast and brightness jitter, noise, scaling etc. have been applied using MATLAB
 R2020a. Although this can be readily done using ImageGenerator, to ensure reproducibility of the results, the augmented
 images are provided in this folder. Post-augmentation, the number of images increased by approximately 14-folds. The
 classes 'Monkeypox' and 'Others' have 1428 and 1764 images, respectively.

3) Fold1: To avoid any sort of bias in training, three-fold cross validation was performed. The original images were split
into training, validation and test set(s) with the approximate proportion of 70 : 10 : 20 while maintaining patient independence. 
According to the commonly perceived data preparation practice, only the training and validation images were augmented while the
test set contained only the original images. Users have the option of using the folds directly or using the original data and
employing other algorithms to augment it.

Additionally, a csv file is provided that has 228 rows and two columns. The table contains the list of all the ImageID(s) with their corresponding label.