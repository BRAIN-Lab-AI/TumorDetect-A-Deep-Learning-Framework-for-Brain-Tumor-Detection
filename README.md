# TumorDetect-A-Deep-Learning-Framework-for-Brain-Tumor-Detection

## Project Metadata
### Authors
- **Team:** Areej Almalki
- **Supervisor Name:** Dr. Muzammil Behzad
- **Affiliations:** IAU and KFUPM

## Introduction
This is a deep learning-based project for brain tumor classification using two complementary approaches: Tucker Decomposition with Neural Networks and Convolutional Neural Networks (CNN). The Tucker decomposition approach is a tensor factorization method that reduces high-dimensional MRI data while preserving spatial structure. It is extremely efficient and achieves high accuracy. The CNN approach has been applied with Grad-CAM visualization to provide interpretable feature maps. However, few studies have examined the combination of dimensionality reduction and explainable AI for brain tumor classification. This is because this domain requires both high accuracy and interpretability simultaneously. This project improves classification by introducing Tucker decomposition for feature extraction and explainable visualization methods including input gradients and Grad-CAM. These approaches allow for efficient processing and interpretable predictions.

## Problem Statement
Deep learning models are among the most accurate classifiers in medical imaging. However, they face some challenges in brain tumor classification tasks. First, high-dimensional MRI data (250x250 pixels) often contains redundant information that increases computational cost and can lead to overfitting. Second, many high-performing models lack interpretability, which makes them hard to use in clinical settings where explanations are needed. Third, subtle visual differences between tumor types (glioma, meningioma, pituitary) also make feature extraction difficult and reduce accuracy.
Thus, the problem is that how to improve classification to detect subtle tumor features while providing visual explanations, and keeping it accurate and efficient. This project solves this by using Tucker decomposition for dimensionality reduction and explainable AI methods to focus better on important features and provide interpretable visualizations.

## Application Area and Project Domain
The project domain includes healthcare, computer vision, and deep learning. The focus is on brain tumor classification using brain MRI scans. The proposed methods are useful in healthcare because they can provide accurate classifications and visual explanations. This helps identify tumor types, support clinical decisions, and build trust through model transparency.

## What is the paper trying to do, and what are you planning to do?
This project implements two approaches for brain tumor classification with explainable visualizations. The first approach uses Tucker decomposition which reduces dimensionality efficiently, and a deep neural network for classification. The second approach uses CNN with Grad-CAM, which provides interpretable class activation maps. These methods help classify tumors with high accuracy (97.33% and 95.73%) and provide visual explanations.
In my project, I plan to explore possible improvements to the models by first understanding how both architectures work, then experimenting with different regularization techniques, comparing visualization methods for interpretability, optimizing training parameters, and analyzing per-class performance to identify strengths of each approach.


### Project Documents
- **Presentation:** [Project Presentation](/presentation.pptx)
- **Report:** [Project Report](/report.pdf)

### Reference Paper
- [Enhanced MRI brain tumor detection and classification via topological data analysis and low-rank tensor decomposition](https://www.sciencedirect.com/science/article/pii/S2772415824000142)

### Reference Dataset
- [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)


## Project Technicalities

### Terminologies
- **Brain Tumor Classification:** Automated classification of MRI images of the brain into tumor type (glioma, meningioma, and pituitary tumor) and no tumor.
- **Medical Imaging:** The use of MRI scans for the visualization of the various structures that are contained within the body.
- **Deep Learning:** Multi-layer neural networks that are capable of learning hierarchical features from data.
- **Neural networks:** Computational models composed of interlinked layers that learn the mappings between inputs and outputs.
- **CNN:** This refers to the architecture of neural networks that utilizes convolutional layers to extract spatial patterns within images.
- **Preprocessing Pipeline:** The operations that are performed on the images before training a model.
- **Tucker Decomposition:** Data dimension reduction with the preservation of spatial information using tensor factorization.
- **Explainable AI:** Techniques that enable the visualization of the features used to drive model predictions.
- **Grad-CAM:** Visualization method emphasizing key parts of the images through the computation of the gradient of the predictions.
- **Saliency Map:** Heatmap of importance of each pixel for predictions using the gradient of the input.
- **Input Gradient:** This is the derivative of the output with respect to the pixels of the input. This illustrates important portions of the input


### Problem Statements
- **Problem 1:** Achieving high-accuracy from high dimentional images remains challenging due to high computational complexity.
- **Problem 2:** There is limited capability in providing explanations about which image regions contribute to model predictions.
- **Problem 3:** Balancing high classification accuracy with model interpretability remains difficult.
  
### Loopholes or Research Areas
- **Evaluation Metrics:** Lack of unified metrics to effectively assess the quality of Interpretability methods (saliency maps, Grad-CAM).
- **Visualization Consistency:** Inconsistencies in explanation quality when applied across different deep learning models.
- **Computational Resources:** Training requires significant GPU compute resources, which may not be readily accessible.

### Problem vs. Ideation: Proposed 3 Ideas to Solve the Problems
1. **Dimensionality Reduction:** Apply Tucker decomposition on the data to reduce the features from 62,500 to 100, improving computational efficiency.
2. **Interpretability:** Use input gradients and Grad-CAM to show which image regions influence predictions.
3. **Dual Architecture:** Implement two complementary methods,Tucker decomposition with fully connected neural networks (97.33% accuracy) and CNN with Grad-CAM (95.73% accuracy), to balance high
   classification accuracy with model interpretability.   

### Proposed Solution: Code-Based Implementation
This repository provides an implementation of brain tumor classification using two approaches. The solution includes:

- **Tucker Decomposition:** Reduces MRI dimensionality while preserving spatial features for efficient neural network training.
- **Deep Neural Network:** Fully connected NN architecture [512, 384, 256, 128] with dropout and L2 regularization to prevent overfitting.
- **CNN with Grad-CAM:** Convolutional architecture with explainable class activation mapping for interpretable predictions.
- **Visualization Methods:** Input gradients and Grad-CAM heatmaps to show which regions influenced classification decisions.

### Key Components
- **`train_classification.py`**: Handles Tucker decomposition, neural network training, and model evaluation..
- **`neural_network_trainer.py`**: Contains neural network architecture and training configuration.
- **`train_detection_nn.py`**: Generates input gradient visualizations and saliency maps.
- **`train_cnn_for_gradcam.py`**: Implements CNN training.
- **`generate_gradcam_only.py`**: Generates Grad-CAM heatmaps for visualization using the trained CNN model.

## Model Workflow
The workflow classify MRI images into tumor types through two approaches:

<img width="960" height="720" alt="dual" src="https://github.com/user-attachments/assets/6f6912b9-a8e4-43a2-802b-1f992108ee90" />

**Method 1:** Tucker Decomposition + Neural Network + Saliency maps
1. **Input:**
  - **Image:** The model takes brain MRI image (250x250 grayscale) as the primary input.
  - **preprocessing:** Gaussian blur, contrast enhancement, normalization.
    
2. **Classification process:** 
  - **Tucker decomposition:** reduces features from 62,500 to 100.
  - **Fully connected neural network:** to classify the tumor type.
3. **Output:**
  - Predicted tumor type with input gradient visualization using Saliency maps.
    
**Method 2:** CNN + Grad-CAM
1. **Input:**
  - Same as Method 1.
2. **Detection process:** 
  - CNN extracts features through 4 layers.
  - Final dence layer to classify the tumor type.
3. **Output:**
  - Predicted tumor type with Grad-CAM heatmap.

<img width="700" height="600" alt="preprocessing_samples" src="https://github.com/user-attachments/assets/94c86fe0-0ab6-4269-8bd2-22f522d6c146" />


## How to Run the Code

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/BRAIN-Lab-AI/TumorDetect-A-Deep-Learning-Framework-for-Brain-Tumor-Detection.git
    cd TumorDetect-A-Deep-Learning-Framework-for-Brain-Tumor-Detection
    ```

2. **Set Up the Environment:**
    Create a virtual environment and install the required dependencies.
    ```bash
      pip install -r requirements.txt
    ```

3. **Prepare Dataset:** Dawnload your brain tumor MRI dataset and organize it in the following structure:
   ```
   TumorDetect-A-Deep-Learning-Framework-for-Brain-Tumor-Detection/
   ├── data/
   │   └── Dataset/
   │       ├── Training/
   │       │   ├── glioma/
   │       │   ├── meningioma/
   │       │   ├── notumor/
   │       │   └── pituitary/
   │       └── Testing/
   │           ├── glioma/
   │           ├── meningioma/
   │           ├── notumor/
   │           └── pituitary/
   ├── src/
   │   ├── train_classification.py
   │   ├── neural_network_trainer.py
   │   ├── train_detection_nn.py
   │   ├── train_cnn_for_gradcam.py
   │   └── generate_gradcam_only.py
   ├── requirements.txt
   └── README.md

   ```
4. **Train Method 1 (Tucker + Neural Network):**
    
    ```bash
    python train_classification.py
    ```

5. **Generate Visualizations using Saliency maps (for Method 1)::**
    
    ```bash
    python train_detection_nn.py
    ```

6. **Train Method 2 (CNN + Grad-CAM):**
    
    ```bash
    python train_cnn_for_gradcam.py
    ```
7. **Generate Grad-CAM Visualizations (for Method 2):**
    
    ```bash
    python generate_gradcam_only.py
    ```
    <img width="1569" height="1076" alt="GRAD_CAM_fig" src="https://github.com/user-attachments/assets/51c8fc9f-3989-439a-a478-34a427e79d03" />

    
## Acknowledgments
- **Open-Source Communities:** Thanks to the contributors of TensorFlow, TensorLy, scikit-learn, and other libraries for their amazing work.
- **Instructor:** Special thanks Dr.Muzammil Behzad for the invaluable guidance and support throughout this project.
- **Dataset:** Thanks to the creators and maintainers of the Brain Tumor MRI Dataset for making this project possible.
