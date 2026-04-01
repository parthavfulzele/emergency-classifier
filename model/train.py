import os
import re
import json
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, accuracy_score

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "dataset.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "classifier.pkl")
METRICS_PATH = os.path.join(os.path.dirname(__file__), "metrics.json")


def preprocess(text):
    """Lowercase, strip punctuation, normalize whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    df = pd.read_csv(DATA_PATH)
    df["text"] = df["text"].apply(preprocess)
    print(f"Loaded {len(df)} samples ({df['label'].sum()} emergency, {(df['label'] == 0).sum()} non-emergency)")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    # Feature union: word n-grams + character n-grams (for typo robustness)
    features = FeatureUnion([
        ("word_tfidf", TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            max_features=10000,
            sublinear_tf=True,
            stop_words="english",
        )),
        ("char_tfidf", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            max_features=10000,
            sublinear_tf=True,
        )),
    ])

    # Try multiple classifiers via GridSearchCV
    # We use a pipeline with the feature union and a classifier placeholder
    pipeline = Pipeline([
        ("features", features),
        ("clf", LogisticRegression()),  # placeholder
    ])

    param_grid = [
        {
            "clf": [LogisticRegression(max_iter=2000, random_state=42)],
            "clf__C": [0.1, 1, 5, 10],
        },
        {
            "clf": [SGDClassifier(loss="modified_huber", max_iter=2000, random_state=42)],
            "clf__alpha": [1e-4, 1e-3, 1e-2],
        },
        {
            "clf": [CalibratedClassifierCV(LinearSVC(max_iter=5000, random_state=42))],
            "clf__estimator__C": [0.1, 1, 5, 10],
        },
    ]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\nRunning GridSearchCV across classifiers...")
    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        verbose=0,
        refit=True,
    )
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    best_clf_name = type(best_model.named_steps["clf"]).__name__
    cv_acc = grid.best_score_

    print(f"\nBest classifier: {best_clf_name}")
    print(f"Best CV accuracy: {cv_acc:.4f}")
    print(f"Best params: {grid.best_params_}")

    # Evaluate on held-out test set
    y_pred = best_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Non-Emergency", "Emergency"])
    report_dict = classification_report(y_test, y_pred, target_names=["Non-Emergency", "Emergency"], output_dict=True)

    print(f"\nTest accuracy: {test_acc:.4f}")
    print(f"\n{report}")

    # Save model
    joblib.dump(best_model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    # Save metrics
    metrics = {
        "accuracy": round(test_acc, 4),
        "cv_accuracy": round(cv_acc, 4),
        "best_classifier": best_clf_name,
        "total_samples": len(df),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "emergency_f1": round(report_dict["Emergency"]["f1-score"], 4),
        "non_emergency_f1": round(report_dict["Non-Emergency"]["f1-score"], 4),
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {METRICS_PATH}")


if __name__ == "__main__":
    main()
