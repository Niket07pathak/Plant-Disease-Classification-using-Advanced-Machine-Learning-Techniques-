
# Plant Disease Classification using Advanced Machine Learning Techniques

## Introduction
This project focuses on developing a machine learning-based solution for classifying plant diseases using images of healthy and damaged leaves. By comparing traditional machine learning models and advanced deep learning architectures, this work aims to identify the most accurate and efficient method for plant disease detection, thus contributing to sustainable agriculture.

## Dataset
The dataset used contains approximately 87,000 high-resolution RGB images of crop leaves, categorized into 38 distinct classes, including both healthy and diseased conditions. It is available publicly:
- **Dataset Link**: [Kaggle Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)

### Data Preprocessing
- Images resized to ensure consistent input dimensions.
- Normalization of pixel values to improve model convergence.
- Techniques like data augmentation, outlier removal, and bootstrap sampling were applied to enhance model performance.

## Models Implemented
The project explores and compares the following machine learning and deep learning models:
1. **Custom CNN**: Designed for efficient feature extraction and high accuracy.
2. **ResNet-18**: Known for handling deep architectures effectively with residual connections.
3. **SVM**: Utilized HOG for feature extraction and classification with RBF kernel.
4. **Random Forest**: Applied ensemble learning for classification.

## Evaluation Metrics
To assess the performance of the models, the following metrics were used:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**

### Model Performance
| Model        | Accuracy  | Precision | Recall  | F1-Score |
|--------------|-----------|-----------|---------|----------|
| ResNet-18    | 98.28%    | 0.98      | 0.98    | 0.98     |
| Custom CNN   | 94.12%    | 0.94      | 0.93    | 0.935    |
| SVM          | 72.23%    | 0.71      | 0.72    | 0.71     |
| Random Forest| 70.85%    | 0.71      | 0.71    | 0.71     |

## Results and Findings
- **Deep learning models**, especially ResNet-18 and Custom CNN, outperformed traditional models in accuracy and efficiency.
- Challenges like overfitting and dataset imbalance were mitigated using regularization, data augmentation, and stratified sampling.
- Explainability techniques, such as Grad-CAM for ResNet-18, provided insights into the decision-making process of the models.

## Future Work
- Deployment of lightweight models for mobile and edge devices.
- Integration of additional data sources, such as soil characteristics and climatic conditions.
- Exploration of ensemble methods for improved accuracy and robustness.
"""
