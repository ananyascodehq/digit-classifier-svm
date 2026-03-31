from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
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
from contextlib import asynccontextmanager

# Global assets
clf = None
scaler = None
analytics_cache = None
eda_cache = None

def load_assets():
    global clf, scaler, analytics_cache
    print("[LOG] Starting asset load...")
    try:
        model_dir = os.path.join(os.getcwd(), 'model')
        clf_path = os.path.join(model_dir, 'svm_model.pkl')
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        
        if os.path.exists(clf_path) and os.path.exists(scaler_path):
            clf = joblib.load(clf_path)
            scaler = joblib.load(scaler_path)
            print("[LOG] Model and Scaler loaded successfully.")
            
            # Precompute analytics and EDA data
            compute_analytics()
            compute_eda()
        else:
            print(f"[ERROR] Model files missing! Looking in: {model_dir}")
            print(f"Contents of model dir: {os.listdir(model_dir) if os.path.exists(model_dir) else 'None'}")
    except Exception as e:
        print(f"[ERROR] Error loading assets: {e}")

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
        print("[LOG] EDA data precomputed.")
    except Exception as e:
        print(f"[ERROR] Error precomputing EDA: {e}")

def compute_analytics():
    global analytics_cache
    if clf is None or scaler is None:
        return
    
    try:
        digits = load_digits()
        X = digits.data
        y = digits.target
        
        pca = PCA(n_components=2)
        X_scaled = scaler.transform(X)
        X_pca = pca.fit_transform(X_scaled)
        
        pca_data = []
        np.random.seed(42) 
        indices = np.random.choice(len(X_pca), min(400, len(X_pca)), replace=False)
        for idx in indices:
            pca_data.append({
                "x": round(float(X_pca[idx][0]), 3),
                "y": round(float(X_pca[idx][1]), 3),
                "label": int(y[idx])
            })
            
        x_points = X_pca[indices, 0]
        y_points = X_pca[indices, 1]
        coeffs = np.polyfit(x_points, y_points, 1)
        line_data = [
            {"x": float(np.min(x_points)), "y": float(coeffs[0] * np.min(x_points) + coeffs[1])},
            {"x": float(np.max(x_points)), "y": float(coeffs[0] * np.max(x_points) + coeffs[1])}
        ]

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
        print("[LOG] Analytics data precomputed.")
    except Exception as e:
        print(f"[ERROR] Error precomputing analytics: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models on startup
    load_assets()
    yield

app = FastAPI(title="ML Digit SVM Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Endpoints
@app.get("/api/health")
async def health():
    return {"status": "ok", "assets_loaded": clf is not None}

@app.get("/api/eda")
async def get_eda_data():
    if eda_cache is None:
        return {"error": "EDA data not available"}
    return eda_cache

@app.get("/api/analytics")
async def get_analytics_data():
    if analytics_cache is None:
        return {"error": "Analytics data not available"}
    return analytics_cache

@app.post("/api/predict-image")
async def predict_image(file: UploadFile = File(...)):
    if clf is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model files are missing on server.")
    try:
        # Re-using the logic from technical report / previous main.py
        img = Image.open(file.file).convert('L')
        img_array = 255 - np.array(img)
        binary_mask = img_array > 50
        struct_size = max(1, img_array.shape[0] // 30)
        struct = np.ones((struct_size, struct_size), dtype=bool)
        dilated = binary_dilation(binary_mask, structure=struct)
        img_array = dilated.astype(np.uint8) * 255
        rows = np.any(img_array > 0, axis=1)
        cols = np.any(img_array > 0, axis=0)
        if not rows.any(): return {"prediction": 0, "status": "empty"}
        rmin, rmax = np.where(rows)[0][[0, -1]]; cmin, cmax = np.where(cols)[0][[0, -1]]
        digit_crop = img_array[rmin:rmax+1, cmin:cmax+1]
        pad = max(digit_crop.shape) // 4
        digit_padded = np.pad(digit_crop, pad_width=pad, mode='constant', constant_values=0)
        pil_resized = Image.fromarray(digit_padded).resize((8, 8), Image.Resampling.LANCZOS)
        img_8x8 = (np.array(pil_resized, dtype=np.float64) / 255.0 * 16.0).reshape(1, -1)
        scaled = scaler.transform(img_8x8)
        prediction = clf.predict(scaled)
        return {"prediction": int(prediction[0]), "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Setup Static Serving
frontend_dist = os.path.join(os.getcwd(), "frontend", "dist")
if os.path.exists(frontend_dist):
    # Serve assets folder
    app.mount("/assets", StaticFiles(directory=os.path.join(frontend_dist, "assets")), name="assets")
    
    # Static files with html=True for SPA (This handles root index.html)
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
else:
    print(f"[WARNING] Static folder not found at {frontend_dist}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
