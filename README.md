# 🐱 Cat Breed Prediction using Deep Learning

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A robust deep learning model capable of classifying cat breeds from images with **88% accuracy**. Built using **Transfer Learning** with **MobileNetV2**, this project not only identifies the breed but also provides detailed breed information such as temperament and origin.

## 📌 Project Overview

Identifying cat breeds can be challenging due to subtle visual differences. This project automates the process using computer vision. By leveraging a pre-trained MobileNetV2 model, we achieved a lightweight and efficient classifier suitable for deployment on mobile or web platforms.

**Key Features:**
* **Multi-class Classification:** Identifies **7 common cat breeds**: *Abyssinian, Bengal, Birman, Bombay, Persian, Ragdoll, and Siamese*.
* **High Accuracy:** Achieved **88.81% validation accuracy** and **98.64% training accuracy**.
* **Detailed Insights:** Returns breed-specific metadata (Temperament, Origin, Description) alongside predictions.
* **Efficient Architecture:** Uses a frozen MobileNetV2 base for fast feature extraction.

## 📂 Dataset

The project utilizes the **Oxford-IIIT Pet Dataset**.
* **Total Images:** ~1,400 images (after filtering for the 7 target breeds).
* **Data Split:**
    * Training Set: 1,111 images
    * Validation Set: 277 images
* **Preprocessing:** Images resized to `224x224` and normalized (`pixel_value / 255.0`).
* **Augmentation:** Random rotation (20°) and horizontal flipping were used to reduce overfitting.

## 🛠️ Technologies Used

* **Python**: Core programming language.
* **TensorFlow / Keras**: For building and training the deep learning model.
* **Pandas**: For managing breed metadata (CSV).
* **NumPy**: For numerical operations.
* **Matplotlib/Seaborn**: For visualizing training performance and confusion matrices.

## 🏗️ Model Architecture

We used **Transfer Learning** to adapt a pre-trained model for this specific task:

1.  **Input Layer:** Accepts `224x224x3` RGB images.
2.  **Base Model:** **MobileNetV2** (pre-trained on ImageNet), with layers **frozen** to act as a feature extractor.
3.  **Pooling:** `GlobalAveragePooling2D` to summarize feature maps into a vector.
4.  **Hidden Layer:** `Dense` layer with **1024 neurons** (ReLU activation) to learn breed-specific patterns.
5.  **Output Layer:** `Dense` layer with **7 neurons** (Softmax activation) for the final probability distribution.

**Optimizer:** Adam | **Loss Function:** Categorical Crossentropy

## 📊 Performance & Results

The model was trained for **10 epochs**.

* **Training Accuracy:** 98.64%
* **Validation Accuracy:** 88.81%

### Confusion Matrix Highlights:
* ✅ **Best Performance:** Birman and Ragdoll breeds.
* ⚠️ **Challenges:** Some confusion observed between *Bombay* and *Abyssinian* breeds due to visual similarities in certain lighting conditions.

## 🚀 Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/cat-breed-prediction.git](https://github.com/your-username/cat-breed-prediction.git)
    cd cat-breed-prediction
    ```

2.  **Install dependencies:**
    ```bash
    pip install tensorflow pandas numpy matplotlib
    ```

3.  **Run the predictor:**
    Update the image path in the script and run:
    ```bash
    python gardio.py
    ```

## 🔮 Future Work

* **Expand Dataset:** Add more images to improve Age Prediction (Kitten vs. Adult), which currently faces data imbalance issues.
* **Fine-tuning:** Unfreeze the top layers of MobileNetV2 to further improve breed classification accuracy.
* **Mobile App:** Deploy the lightweight model into a Flutter or React Native app for real-time identification.

