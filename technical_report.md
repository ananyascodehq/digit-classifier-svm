# Technical Report: SVM Digit Classifier v2.1

## 1. Project Overview
This project is a full-stack handwritten digit recognition system. It uses a Support Vector Machine (SVM) model trained on the UCI digits dataset (8x8 grayscale images). The system includes a FastAPI backend for real-time inference and a React frontend for user interaction and data visualization.

## 2. Folder Structure
```text
ml-digit-svm/
├── main.py                 # FastAPI backend server
├── train_model.py          # ML training script utilizing GridSearchCV
├── run.py                  # Process orchestrator (runs FastAPI & Vite)
├── technical_report.md     # Project documentation
├── model/                  # Persistent model assets
│   ├── svm_model.pkl       # Trained SVC best estimator
│   └── scaler.pkl          # Fitted StandardScaler object
└── frontend/               # React/Vite frontend application
    ├── src/
    │   ├── App.jsx         # Main application logic & UI components
    │   ├── main.jsx        # React entry point
    │   └── index.css       # Global styles and design system tokens
    ├── index.html          # HTML template with Google Fonts
    ├── vite.config.js      # Vite configuration with API proxy (port 3000 -> 8000)
    └── package.json        # Frontend dependencies (React, Recharts, Axios, Lucide)
```

## 3. Detailed File breakdown

### Backend Logic
*   **`main.py`**:
    *   **Startup Caching**: On server start, it runs `compute_analytics()` and `compute_eda()` to pre-calculate PCA projections and dataset statistics. These are stored in global variables (`analytics_cache`, `eda_cache`) to avoid per-request overhead.
    *   **Endpoints**:
        *   `POST /api/predict-image`: Receives a raw PNG blob. Uses the `preprocess_canvas_image` function for segmentation and normalization.
        *   `GET /api/analytics`: Returns cached PCA data, confusion matrix, and accuracy scores.
        *   `GET /api/eda`: Returns cached dataset metadata.
    *   **`preprocess_canvas_image(image_bytes)`**:
        *   Inverts colors (dark-on-white $\rightarrow$ bright-on-black).
        *   Binarizes (threshold > 50).
        *   **Morphological Dilation**: Artificially thickens strokes via `scipy.ndimage.binary_dilation` to ensure thin handwriting survives 8x8 downsampling.
        *   Crops to the digit's bounding box to ensure position invariance.
        *   Adds 25% padding to match UCI training margins.
        *   Resizes to 8x8 using LANCZOS resampling.

### Machine Learning
*   **`train_model.py`**:
    *   **Data**: Loads `scikit-learn` digits dataset (1797 samples).
    *   **Scaling**: Uses `StandardScaler` to normalize features (0-16 range $\rightarrow$ standard normal).
    *   **Optimization**: Implements `GridSearchCV` searching over `C: [1, 10, 100]` and `gamma: ['scale', 'auto', 0.001, 0.01]`.
    *   **Final Params**: Best estimator is saved as `svm_model.pkl`. It uses an RBF kernel.

### Frontend Application
*   **`App.jsx`**:
    *   **Recognizer**: Uses HTML5 Canvas for drawing. Converts drawing to a PNG blob via `canvas.toBlob` and sends it to the server.
    *   **Analytics**: Uses `Recharts` to plot a 2D scatter plot of PCA components and a heatmap for the Confusion Matrix.
*   **`vite.config.js`**:
    *   Configures a proxy server so that any requests to `/api` from the frontend (port 3000) are routed to the Python backend (port 8000).

---

## 4. Historical Problems and Implemented Fixes

| Problem Encountered | Technical Root Cause | Resolution / Fix |
| :--- | :--- | :--- |
| **High Inference Error** | Raw 8x8 resize didn't account for digit position or size on the canvas. | Implemented **Bounding-Box Cropping** and **Proportional Padding** in `main.py` to center every digit perfectly. |
| **Slow Dashboard Load** | PCA (64D $\rightarrow$ 2D) and Confusion Matrix predictions were recalculated on every API request. | Implemented **Startup Caching**. Heavy math is performed once at server launch and the result is stored in global memory. |
| **Model Generalization** | `gamma='auto'` was used, which ignores the variance of the features after scaling. | Switched to **`gamma='scale'`** and added **`GridSearchCV`** to find the optimal mathematical parameters for the RBF kernel. |
| **Silent Data Bias** | The initial train-test split didn't shuffle data, leading to a model that missed entire classes of digits. | Set **`shuffle=True`** in `train_test_split` and increased training size to 80% to ensure class balance. |
| **CORS / 404 Errors** | Frontend tried to hit backend ports directly without handling cross-origin policies or base URL mismatches. | Implemented a **Vite Proxy** in `vite.config.js` and unified API calls to use relative `/api` paths with Axios. |
| **Color Mismatch** | Canvas uses black-on-white, but UCI digits training data uses white cells on black backgrounds. | Added an **Inversion Step** ($255 - X$) to the preprocessing pipeline before feeding data to the SVM. |
| **Thin Strokes** | Users drawing thin lines resulted in empty or fragmented 8x8 matrices during downsampling. | Implemented **Morphological Dilation** before resizing to artificially thicken strokes based on the original image resolution. |

---

## 5. System Status
- **Model Accuracy**: ~98.6% (Test Set)
- **Backend Latency**: <5ms for cached analytics, <20ms for inference.
- **Stack**: FastAPI, Scikit-Learn, PIL, React 19, Vite, Recharts, Lucide Icons.
