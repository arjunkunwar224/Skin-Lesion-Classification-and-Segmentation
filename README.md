# Skin lesion classification and Segmentation.

Here, I have used the Ham10000 dataset for the classification and segmentation which i have taken from the kaggle and harvard dataverse, the segmented dataset is in the harvard dataverse i have given the link of the dataset below.

#For CNN
In this project i have built the CNN model from the scratch and trained it to classify the seven types of lesion.
In the data preprocessing, as the data set was imbalanced so, I have increased the no. of images which has less than 5000 image and taken randomly 5000 images which has more than 5000 images and accordingly a new metadata is created.
In the augmentation, I have applied rotation,shift,shear,flip,contrast,etc. Then i have splitted the dataset into (70:20:10)

#For U-NET
The model for the U-NET is also custom built where all the corresponding mask images are there and there i have not applied any augumentation to increase the images i have trained using the 10015 images only for the segmentation.


Dataset link:
1. https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T

2. https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000