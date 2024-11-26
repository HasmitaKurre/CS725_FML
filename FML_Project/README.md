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
* [Presentation](https://docs.google.com/presentation/d/1qWaXAnS5kQA2f15G3cFvWv3OZH8XOr8yN9_NEfvfiEM/view?usp=sharing)

## Problem Statement

> Develop a system for soil moisture detection to accurately predict soil moisture levels.

* We performed data preprocessing and sampling techniques to fine tune the data.
* For this project we had explored ML algorithms for Image Classification like CNN,
ResNet, AlexNet and MobileNet to detect the moisture and compared these
algorithms by calculating their accuracy and f1-score.

## Dataset

### Dataset Generation

We used satellite data from NASA-USDA on SMAP soil moisture profile provided using Google Maps API.
We saved snapshots from 2015 to 2020 and clipped the region of India.
We created patches of `396 x 396` dimensions of the region and assigned groundtruth label according to the intensity levels provided by the profiler.
Patches of a groundtruth label are stored in the directory named as that class label.

### Dataset Sampling

![Dataset class-wise #patches](./assets/dataset_class_distribution.png)

Due to such uneven #patches we selected three classes Dry, Moderate_Moisture and Highest_Moisture with considerable images to train on. Also, we had to downsample Highest_Moisture and upsample Moderate_Moisture class patches.

The following table shows the final dataset stats:

|Moisture Class|#Patches|
|:---:|:---:|
|Dry|5001|
|Moderate_Moisture|5001|
|Highest_Moisture|5000|

### Dataset Masking

The pixel coloring around Indian borders were white which created possibility of classifier to classify these images in meaningless class where patches have dominating white regions.
So, we masked(recolored) those regions with maximum amount of color intensity.

> The following diagram can help better visualize overall process:

![Dataset Generation](./assets/dataset_processing.png)

### Dataset Split

* Training split: 70%
* Validation split: 20%

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

> Launch `jupyter server` on this directory
> Run the notebooks present in `src` folder in following order:

  1. Run [`1_Get_NASA_Raw_Data_FML.ipynb`](./code/1_Get_NASA_Raw_Data_FML.ipynb) jupyter notebook.
  2. Run [`2_Visualize_NASA_Raw_Data_FML.ipynb`](./code/2_Visualize_NASA_Raw_Data_FML.ipynb) jupyter notebook.
  3. Run [`3_Visualize_Images_NASA_Image_Data_FML.ipynb`](./code/3_Visualize_Images_NASA_Image_Data_FML.ipynb) jupyter notebook.
  4. Run [`4_Visualize_Images_NASA_Patch_Data_FML.ipynb`](./code/4_Visualize_Images_NASA_Patch_Data_FML.ipynb) jupyter notebook.
  5. Run [`5_Sampling_Classes.ipynb`](./code/5_Sampling_Classes.ipynb) jupyter notebook.
  6. Run [`6_Preprocesing_Classes.ipynb`](./code/6_Preprocesing_Classes.ipynb) jupyter notebook.
  7. Run [`7_CNN.ipynb`](./code/7_CNN.ipynb) jupyter notebook.
  8. Run [`8_Alexnet.ipynb`](./code/8_Alexnet.ipynb) jupyter notebook.
  9. Run [`9_MonbileNet.ipynb`](./code/9_MonbileNet.ipynb) jupyter notebook.
  10. Run [`10_ResNet50.ipynb`](./code/10_ResNet50.ipynb) jupyter notebook.

> You can find the results of Confusion matrix, Accuracy vs loss plot, Actual vs Prediction plot in `results/` folder

## Results

### Observations

Following images are some inference tests using trained CNN model:

![Trained CNN model Inference Results](./assets/inference_results.png)

* All classes are classified perfectly.
* Performance is consistent across each class.
* Model performs very well in classifying images without notable errors.
* Minimal misclassifications and good predictions.

This is the confusion matrix of the CNN model trained.

![Confusion Matrix of trained CNN Model](./assets/cnn_confusion_matrix.png)

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
