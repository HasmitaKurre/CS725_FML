# Soil Moisture Detection and Prediction

## Academic Programme

Foundations of Machine Learning (CS-725)

## Team Details

1. Hasmita Kurre : 23D0385
2. Chaitanya Shinge : 23M2116
3. Chanakya Vihar Challa : 24M2028
4. Rohit Kumar : 24M2029

## Team Contribution

* Hasmita Kurre
  * Dataset Preparation
  * CNN implementation on the dataset.

* Rohit Kumar
  * AlexNet implementation on the dataset.

* Chaitanya Shinge
  * ResNet50 implementation on the dataset.

* Chanakya Vihar Challa
  * MobileNetV2 implementation on the dataset.

## Submission

* [GitHub Code](https://github.com/HasmitaKurre/CS725_FML/tree/main/FML_Project)
* [Soil Moisture Dataset](https://www.kaggle.com/datasets/hasmitakurre/nasa-soil-moisture-india-dataset)

## Problem Statement

Develop a system for soil moisture detection to accurately predict soil moisture levels.

* We performed data preprocessing and sampling techniques to fine tune the data.
* For this project we had explored ML algorithms for Image Classification like CNN,
ResNet, AlexNet and MobileNet to detect the moisture and compared these
algorithms by calculating their accuracy and f1-score.

## Dataset

NASA-USDA, SMAP soil moisture profile satellite data from 2015 to 2020 clipped for India region

* Patch Dimensions: 396 x 396
* Training split: 70%
* Validation split: 20%

|Moisture Class|#Patches|
|:---:|:---:|
|Dry|5001|
|Moderate_Moisture|5001|
|Highest_Moisture|5000|

## Techniques Used

### CNN

### AlexNet

### MobileNetV2

### ResNet50

* Common Hyperparameters across Models
  * Loss functions Image Size: 96 x 96 x 3
  * Epochs: 200 (but also using Early Stopping)
  * Training Batch Size: 32
* Calculated Accuracy, F1-score, Recall and Precision using scikit-learn

## Run

1. Launch `jupyter server` on this directory
2. Run the notebooks present in Code folder in following order

  1. Run " 1_Get_NASA_Raw_Data_FML.ipynb" jupyter notebook.
  2. Run " 2_Visualize_NASA_Raw_Data_FML.ipynb" jupyter notebook.
  3. Run " 3_Visualize_Images_NASA_Image_Data_FML.ipynb" jupyter notebook.
  4. Run " 4_Visualize_Images_NASA_Patch_Data_FML.ipynb" jupyter notebook.
  5. Run " 5_Sampling_Classes.ipynb" jupyter notebook.
  6. Run " 6_Preprocesing_Classes.ipynb" jupyter notebook.
  7. Run " 7_CNN.ipynb" jupyter notebook.
  8. Run " 8_Alexnet.ipynb" jupyter notebook.
  9. Run " 9_MonbileNet.ipynb" jupyter notebook.
  10. Run " 10_ResNet50.ipynb" jupyter notebook.

You can find the results of Confusion matrix, Accuracy vs loss plot,
Actual vs Prediction plot, ROC-AUC curve in `results/` folder

## Results

### Observation

* All classes are classified perfectly.
* Performance is consistent across each class.
* Model performs very well in classifying images without notable errors.
* Minimal misclassifications and good predictions.

|Models|Test Accuracy|
|:---:|:---:|
|CNN|99.57%|
|ResNet50|99.66%|
|MobileNetV2|98.71%|
|AlexNet|97.42%|

## Conclusion

Our main aim is soil moisture detection to accurately predict soil moisture levels using
various machine learning algorithms and compare their accuracy to find the best
algorithm for the soil moisture detection.
