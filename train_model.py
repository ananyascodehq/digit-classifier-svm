import joblib
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import os

def train_model():
    print("Loading digits dataset...")
    digits = datasets.load_digits()
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))

    print(f"Splitting data into training and test sets (n_samples={n_samples})...")
    X_train, X_test, y_train, y_test = train_test_split(
        data, digits.target, test_size=0.2, shuffle=True, random_state=42
    )

    print("Standardizing features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Use GridSearchCV for hyperparameter optimization
    print("Initializing GridSearchCV for SVM optimization...")
    param_grid = {
        'C': [1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01],
        'kernel': ['rbf']
    }
    
    grid_search = GridSearchCV(
        svm.SVC(), 
        param_grid, 
        cv=5, 
        n_jobs=-1, 
        verbose=1
    )

    print("Training model with GridSearchCV...")
    grid_search.fit(X_train, y_train)
    
    clf = grid_search.best_estimator_
    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

    print("Evaluating model...")
    predicted = clf.predict(X_test)
    print(f"Classification report:\n{metrics.classification_report(y_test, predicted)}")

    # Ensure model directory exists
    os.makedirs('model', exist_ok=True)
    
    print("Saving model and scaler to 'model/'...")
    joblib.dump(clf, 'model/svm_model.pkl')
    joblib.dump(scaler, 'model/scaler.pkl')
    print("Model and scaler saved successfully.")

if __name__ == "__main__":
    train_model()
