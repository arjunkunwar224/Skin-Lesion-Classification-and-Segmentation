# Skin lesion classification and Segmentation.
In this project, I have used the HAM10000 dataset for both skin lesion classification and segmentation. The dataset was sourced from Kaggle and the Harvard Dataverse. Specifically, the segmentation masks are available through the Harvard Dataverse, and I have included the dataset link below.

🔹 **Classification Using CNN**

For the classification task, I built a Convolutional Neural Network (CNN) from scratch to classify skin lesions into seven categories. Since the dataset was imbalanced, I applied class balancing strategies:
-> For classes with fewer than 5,000 images, I increased the number using augmentation.
-> For classes with more than 5,000 images, I randomly selected 5,000 samples. Based on this, a new metadata file was generated.

->For data augmentation, I applied techniques such as rotation, shifting, shearing, flipping, and contrast adjustments. The dataset was then split into training, validation, and testing sets in a 70:20:10 ratio.

🔹 **Segmentation Using U-Net**

For the segmentation task, I also built a custom U-Net model. The dataset includes corresponding mask images for each input image. In this case, no additional augmentation was performed. The model was trained using 10,015 paired images.


**Dataset link:**
1. https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T

2. https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000
