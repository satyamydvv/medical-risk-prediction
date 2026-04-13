from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import joblib
import matplotlib.pyplot as plt
import numpy as np


# 🔥 TRAIN MODEL WITH HYPERPARAMETER TUNING
def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [5, 10, 15],
        "min_samples_split": [2, 5, 10]
    }

    grid = GridSearchCV(
        model,
        param_grid,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    print("Best Parameters:", grid.best_params_)

    return grid.best_estimator_


# 📊 EVALUATE MODEL
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nROC-AUC Score:", roc_auc_score(y_test, y_prob))


# 📈 FEATURE IMPORTANCE (EXPLAINABILITY)
def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure()
    plt.title("Feature Importance")
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), feature_names[indices], rotation=90)
    plt.tight_layout()
    
    plt.savefig("models/feature_importance.png")
    print("Feature importance plot saved in models/")


# 💾 SAVE MODEL
def save_model(model, scaler):
    joblib.dump(model, "models/random_forest.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
