# TumorDetect-A-Deep-Learning-Framework-for-Brain-Tumor-Detection

## Project Metadata
### Authors
- **Team:** Areej Almalki
- **Supervisor Name:** Dr. Muzammil Behzad
- **Affiliations:** IAU and KFUPM

## Introduction
This is a deep learning–based project for brain tumor detection using a well-known object detection model called YOLO (You Only Look Once). YOLO is a real-time system for object detection in images and videos. It is extremely fast and achieves high accuracy. The model has been applied to many fields such as autonomous driving and healthcare. However, few studies have examined the effectiveness of YOLO networks in brain tumor detection. This is because this domain requires both high precision and low latency simultaneously. The RCS-YOLO paper improves the YOLO model by introducing the Reparameterized Convolution with Channel Shuffle (RCS) module and One-Shot Aggregation (OSA) of RCS blocks. These novel improvements allow for richer feature extraction and reduced time consumption.

## Problem Statement
YOLO models are among the fastest object detectors in deep learning. However, they face some challenges in medical imaging tasks such as brain tumor detection. First, detecting small or low-contrast tumors is difficult because standard convolutional layers often lose fine details as the network goes deeper. Second, many high-performing models need heavy computation, which makes them hard to use in real-time medical problems. Third, feature redundancy and inefficient aggregation of multi-scale information also waste resources and reduce accuracy.
Thus, the problem is that how to improve YOLO to detect subtle tumor features, such as small or low-contrast anomalies, while keeping it fast and efficient. RCS-YOLO solves this by changing the convolution blocks and aggregation methods to focus better on important channels and spatial details.

## Application Area and Project Domain
The project domain includes healthcare, computer vision, and deep learning. The focus is on brain tumor detection using brain MRI scans. The proposed model is useful in healthcare because it can provide accurate detections and bounding boxes. This helps identify tumors, plan surgeries, and track treatment progress.

## What is the paper trying to do, and what are you planning to do?
The RCS-YOLO paper improves YOLO for brain tumor detection by adding two main ideas: the RCS module, which makes feature extraction more efficient, and the RCS-OSA block, which combines features more effectively. These changes help the model detect tumors with better accuracy and faster speed. 
In my project, I plan to explore possible improvements to the model by first understanding how the RCS-YOLO architecture works, then experimenting with different loss functions, applying regularization techniques, optimizing training parameters, and making small modifications to the layers.

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
- **Diffusion Model:** A generative model that progressively transforms random noise into coherent data.
- **Latent Space:** A compressed, abstract representation of data where complex features are captured.
- **UNet Architecture:** A neural network with an encoder-decoder structure featuring skip connections for better feature preservation.
- **Text Encoder:** A model that converts text into numerical embeddings for downstream tasks.
- **Perceptual Loss:** A loss function that measures high-level differences between images, emphasizing perceptual similarity.
- **Tokenization:** The process of breaking down text into smaller units (tokens) for processing.
- **Noise Vector:** A randomly generated vector used to initialize the diffusion process in generative models.
- **Decoder:** A network component that transforms latent representations back into image space.
- **Iterative Refinement:** The process of gradually improving the quality of generated data through multiple steps.
- **Conditional Generation:** The process where outputs are generated based on auxiliary inputs, such as textual descriptions.

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
