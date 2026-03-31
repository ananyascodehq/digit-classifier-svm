---
title: SVM Digit Classifier
emoji: 🔢
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: true
license: mit
---

# 🔢 SVM Digit Classifier

A high-performance, real-time handwritten digit recognition system built with **FastAPI**, **React**, and **Scikit-Learn**. This application uses a Support Vector Machine (SVM) to classify hand-drawn digits with ~98% accuracy.

![Preview](https://raw.githubusercontent.com/ananyascodehq/digit-classifier-svm/main/screenshot.png) *(Note: Add your own screenshot path here)*

## 🚀 Live Demo
Experience the real-time classifier on **Hugging Face Spaces**: [View Space](https://huggingface.co/spaces/ananyakannan/digit-classifier-svm)

---

## ✨ Key Features

-   **🖌️ Interactive Canvas**: Draw digits directly on the screen for instant, real-time predictions.
-   **📊 Analytics Dashboard**:
    -   **PCA Projection**: A 2D visualization of the high-dimensional (64D) feature space.
    -   **Confusion Matrix**: Deep dive into model performance across all 10 digits.
    -   **Data Distribution**: Statistics on the underlying UCI Digits dataset.
-   **🧠 Advanced Preprocessing**:
    -   **Morphological Dilation**: Ensures thin handwriting survives 8x8 downsampling.
    -   **Bounding-Box Cropping**: Position and size-invariant recognition.
    -   **LANCZOS Resampling**: High-quality image reduction for model input.
-   **⚡ Optimized Performance**:
    -   **Startup Caching**: Heavy analytical computations are pre-calculated at server launch.
    -   **Vite Managed Frontend**: Instant hot-reloading and lightning-fast production builds.

---

## 📂 Dataset: UCI Optical Recognition of Handwritten Digits

This project utilizes the **UCI Digits Dataset** (available via `scikit-learn`):
-   **Size**: 1,797 samples.
-   **Classes**: 10 (digits 0-9).
-   **Dimensions**: 8x8 pixels (grayscale, 0-16 intensity range).
-   **Balanced**: Roughly 180 samples per class, ensuring a non-biased training process.

---

## 🛠️ Technical Approach

### 1. The Model (SVM)
We use a **Support Vector Classifier (SVC)** with a **Radial Basis Function (RBF) kernel**. The model hyperparameters ($C$ and $\gamma$) were optimized using `GridSearchCV` to achieve peak generalization:
-   **Kernel**: RBF
-   **C**: 10
-   **Gamma**: 'scale' (optimized for feature variance)

### 2. The Inference Pipeline
When a user draws a digit, the following transformations occur:
1.  **Inversion**: Converts canvas colors (black-on-white) to the model's expected format (bright-on-black).
2.  **Dilation**: Thickens strokes using `scipy.ndimage` to prevent signal loss during compression.
3.  **Segmentation**: Crops the image to the exact bounding box of the strokes to eliminate white space bias.
4.  **Standardization**: Applies a `StandardScaler` fitted on the original training distribution.

### 3. Dimensionality Reduction
To visualize the 64-dimensional pixel data on a 2D dashboard, we use **Principal Component Analysis (PCA)**. This preserves the maximum possible variance and shows how the SVM separates different digit clusters in space.

---

## 📦 Installation & Local Setup

### Using the Process Orchestrator (Recommended)
1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/yourusername/ml-digit-svm.git
    cd ml-digit-svm
    ```
2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    cd frontend && npm install && cd ..
    ```
3.  **Run Everything**:
    ```bash
    python run.py
    ```
    *This will start the FastAPI backend (port 8000), the Vite frontend (port 3000), and open your browser automatically.*

### Using Docker
```bash
docker build -t digit-svm .
docker run -p 8000:8000 digit-svm
```

---

## 🤝 Project Structure
```text
ml-digit-svm/
├── main.py                 # FastAPI backend & production server
├── train_model.py          # Model training & optimization script
├── run.py                  # Local development orchestrator
├── model/                  # Trained artifacts (.pkl)
└── frontend/               # React application source
```

## 📜 License
This project is licensed under the MIT License.
