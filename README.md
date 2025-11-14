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

# THE FOLLOWING IS SUPPOSED TO BE DONE LATER

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
- **Problem 1:** Achieving high-resolution and detailed images using conventional diffusion models remains challenging.
- **Problem 2:** Existing models suffer from slow inference times during the image generation process.
- **Problem 3:** There is limited capability in performing style transfer and generating diverse artistic variations.

### Loopholes or Research Areas
- **Evaluation Metrics:** Lack of robust metrics to effectively assess the quality of generated images.
- **Output Consistency:** Inconsistencies in output quality when scaling the model to higher resolutions.
- **Computational Resources:** Training requires significant GPU compute resources, which may not be readily accessible.

### Problem vs. Ideation: Proposed 3 Ideas to Solve the Problems
1. **Optimized Architecture:** Redesign the model architecture to improve efficiency and balance image quality with faster inference.
2. **Advanced Loss Functions:** Integrate novel loss functions (e.g., perceptual loss) to better capture artistic nuances and structural details.
3. **Enhanced Data Augmentation:** Implement sophisticated data augmentation strategies to improve the model’s robustness and reduce overfitting.

### Proposed Solution: Code-Based Implementation
This repository provides an implementation of the enhanced stable diffusion model using PyTorch. The solution includes:

- **Modified UNet Architecture:** Incorporates residual connections and efficient convolutional blocks.
- **Novel Loss Functions:** Combines Mean Squared Error (MSE) with perceptual loss to enhance feature learning.
- **Optimized Training Loop:** Reduces computational overhead while maintaining performance.

### Key Components
- **`model.py`**: Contains the modified UNet architecture and other model components.
- **`train.py`**: Script to handle the training process with configurable parameters.
- **`utils.py`**: Utility functions for data processing, augmentation, and metric evaluations.
- **`inference.py`**: Script for generating images using the trained model.

## Model Workflow
The workflow of the Enhanced Stable Diffusion model is designed to translate textual descriptions into high-quality artistic images through a multi-step diffusion process:

1. **Input:**
   - **Text Prompt:** The model takes a text prompt (e.g., "A surreal landscape with mountains and rivers") as the primary input.
   - **Tokenization:** The text prompt is tokenized and processed through a text encoder (such as a CLIP model) to obtain meaningful embeddings.
   - **Latent Noise:** A random latent noise vector is generated to initialize the diffusion process, which is then conditioned on the text embeddings.

2. **Diffusion Process:**
   - **Iterative Refinement:** The conditioned latent vector is fed into a modified UNet architecture. The model iteratively refines this vector by reversing a diffusion process, gradually reducing noise while preserving the text-conditioned features.
   - **Intermediate States:** At each step, intermediate latent representations are produced that increasingly capture the structure and details dictated by the text prompt.

3. **Output:**
   - **Decoding:** The final refined latent representation is passed through a decoder (often part of a Variational Autoencoder setup) to generate the final image.
   - **Generated Image:** The output is a synthesized image that visually represents the input text prompt, complete with artistic style and detail.

## How to Run the Code

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/yourusername/enhanced-stable-diffusion.git
    cd enhanced-stable-diffusion
    ```

2. **Set Up the Environment:**
    Create a virtual environment and install the required dependencies.
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3. **Train the Model:**
    Configure the training parameters in the provided configuration file and run:
    ```bash
    python train.py --config configs/train_config.yaml
    ```

4. **Generate Images:**
    Once training is complete, use the inference script to generate images.
    ```bash
    python inference.py --checkpoint path/to/checkpoint.pt --input "A surreal landscape with mountains and rivers"
    ```

## Acknowledgments
- **Open-Source Communities:** Thanks to the contributors of PyTorch, Hugging Face, and other libraries for their amazing work.
- **Individuals:** Special thanks to bla, bla, bla for the amazing team effort, invaluable guidance and support throughout this project.
- **Resource Providers:** Gratitude to ABC-organization for providing the computational resources necessary for this project.
