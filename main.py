from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
from PIL import Image
import io
import os
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from scipy.ndimage import binary_dilation

app = FastAPI(title="ML Digit SVM Backend")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global assets
clf = None
scaler = None
analytics_cache = None
eda_cache = None

def load_assets():
    global clf, scaler, analytics_cache
    try:
        model_dir = os.path.join(os.getcwd(), 'model')
        clf_path = os.path.join(model_dir, 'svm_model.pkl')
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        
        if os.path.exists(clf_path) and os.path.exists(scaler_path):
            clf = joblib.load(clf_path)
            scaler = joblib.load(scaler_path)
            print("Model and Scaler loaded successfully.")
            
            # Precompute analytics and EDA data
            compute_analytics()
            compute_eda()
    except Exception as e:
        print(f"Error loading assets: {e}")

def compute_eda():
    global eda_cache
    try:
        digits = load_digits()
        unique, counts = np.unique(digits.target, return_counts=True)
        class_distribution = [{"digit": int(u), "count": int(c)} for u, c in zip(unique, counts)]
        
        eda_cache = {
            "total_samples": int(len(digits.target)),
            "class_distribution": class_distribution,
            "resolution": "8x8",
            "feature_range": [0, 16]
        }
        print("EDA data precomputed and cached.")
    except Exception as e:
        print(f"Error precomputing EDA: {e}")

def compute_analytics():
    global analytics_cache
    if clf is None or scaler is None:
        return
    
    try:
        digits = load_digits()
        X = digits.data
        y = digits.target
        
        # 1. PCA Projection (2D)
        pca = PCA(n_components=2)
        X_scaled = scaler.transform(X)
        X_pca = pca.fit_transform(X_scaled)
        
        pca_data = []
        # Use a fixed random seed for consistent sampling in the cache
        np.random.seed(42) 
        indices = np.random.choice(len(X_pca), min(400, len(X_pca)), replace=False)
        for idx in indices:
            pca_data.append({
                "x": round(float(X_pca[idx][0]), 3),
                "y": round(float(X_pca[idx][1]), 3),
                "label": int(y[idx])
            })
            
        # 2. Decision Boundary (Fitted Line equivalent for PCA)
        x_points = X_pca[indices, 0]
        y_points = X_pca[indices, 1]
        coeffs = np.polyfit(x_points, y_points, 1)
        line_data = [
            {"x": float(np.min(x_points)), "y": float(coeffs[0] * np.min(x_points) + coeffs[1])},
            {"x": float(np.max(x_points)), "y": float(coeffs[0] * np.max(x_points) + coeffs[1])}
        ]

        # 3. Confusion Matrix
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)
        X_test_scaled = scaler.transform(X_test)
        y_pred = clf.predict(X_test_scaled)
        cm = confusion_matrix(y_test, y_pred)
        
        cm_data = []
        for i in range(10):
            for j in range(10):
                cm_data.append({
                    "actual": i,
                    "predicted": j,
                    "value": int(cm[i][j])
                })

        analytics_cache = {
            "pca_data": pca_data,
            "line_data": line_data,
            "confusion_matrix": cm_data,
            "accuracy": round(float(clf.score(X_test_scaled, y_test)), 4)
        }
        print("Analytics data precomputed and cached.")
    except Exception as e:
        print(f"Error precomputing analytics: {e}")

load_assets()

@app.get("/")
def read_root():
    return {"message": "SVM Digit Classifier API is running!"}

@app.get("/api/eda")
def get_eda_data():
    if eda_cache is None:
        compute_eda()
        if eda_cache is None:
            return {"error": "EDA data not available"}
    return eda_cache

@app.get("/api/analytics")
def get_analytics_data():
    if analytics_cache is None:
        # If cache is missing (e.g. error at startup), try to compute it once
        compute_analytics()
        if analytics_cache is None:
            return {"error": "Analytics data not available"}
    
    return analytics_cache

def preprocess_canvas_image(image_bytes):
    """
    Replicates the preprocessing the UCI digits dataset applies.
    Input:  raw image bytes from canvas (black ink on white bg)
    Output: (1, 64) normalized array ready for clf.predict()
    """
    # 1. Load and convert to grayscale
    img = Image.open(image_bytes).convert('L')
    img_array = np.array(img)

    # 2. Invert: canvas is dark-on-white, training data is bright-on-black
    img_array = 255 - img_array

    # 3. Threshold: remove noise, make strokes binary
    binary_mask = img_array > 50
    
    # 3.5 ── Morphological Dilation ──
    # Artificially thicken strokes before cropping/resizing to ensure signals 
    # survive downsampling to 8x8.
    struct_size = max(1, img_array.shape[0] // 30)
    struct = np.ones((struct_size, struct_size), dtype=bool)
    dilated = binary_dilation(binary_mask, structure=struct)
    img_array = dilated.astype(np.uint8) * 255

    # 4. ── CRITICAL: crop to bounding box of the drawn digit ──
    rows = np.any(img_array > 0, axis=1)
    cols = np.any(img_array > 0, axis=0)

    if not rows.any():
        # Empty canvas — return zeros
        return np.zeros((1, 64))

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    digit_crop = img_array[rmin:rmax+1, cmin:cmax+1]

    # 5. Add padding proportional to the crop size (mimics dataset margins)
    pad = max(digit_crop.shape) // 4
    digit_padded = np.pad(digit_crop, pad_width=pad, mode='constant', constant_values=0)

    # 6. Resize to 8x8
    pil_cropped = Image.fromarray(digit_padded)
    pil_resized = pil_cropped.resize((8, 8), Image.Resampling.LANCZOS)
    img_8x8 = np.array(pil_resized, dtype=np.float64)

    # 7. Rescale to 0–16 range to match training distribution
    img_8x8 = img_8x8 / 255.0 * 16.0

    # 8. Apply the SAME scaler fitted during training
    pixel_array = img_8x8.reshape(1, -1)
    return pixel_array

@app.post("/api/predict")
async def predict(data: dict):
    if clf is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model and Scaler must be loaded")
    try:
        pixel_data = np.array(data['image']).reshape(1, -1)
        scaled_data = scaler.transform(pixel_data)
        prediction = clf.predict(scaled_data)
        return {"prediction": int(prediction[0]), "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/predict-image")
async def predict_image(file: UploadFile = File(...)):
    if clf is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model and Scaler must be loaded.")
    try:
        raw = preprocess_canvas_image(file.file)
        scaled = scaler.transform(raw)
        prediction = clf.predict(scaled)
        return {"prediction": int(prediction[0]), "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
