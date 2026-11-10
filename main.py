import argparse
from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

TARGET = "creditworthy"

def make_demo_data(path, rows=1200, seed=42):
    """Create a clearly-labelled synthetic dataset so the project runs immediately."""
    import numpy as np
    rng = np.random.default_rng(seed)
    income = rng.integers(18000, 180001, rows)
    debt = rng.integers(0, 90001, rows)
    age = rng.integers(18, 71, rows)
    late = rng.integers(0, 8, rows)
    employment = rng.choice(["salaried", "self_employed", "unemployed"], rows,
                            p=[0.62, 0.28, 0.10])
    credit_history = rng.integers(0, 21, rows)
    score = (income / 10000) - (debt / 12000) - (late * 1.35) + \
            (credit_history * 0.35) + (age >= 25) * 1.2 + \
            (employment == "salaried") * 1.5 + rng.normal(0, 2.2, rows)
    creditworthy = (score > 8.5).astype(int)
    df = pd.DataFrame({
        "income": income, "debt": debt, "age": age,
        "late_payments": late, "employment": employment,
        "credit_history_years": credit_history, TARGET: creditworthy
    })
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Created synthetic demo data: {path}")

def train(data_path, model_name="random_forest"):
    df = pd.read_csv(data_path)
    if TARGET not in df.columns:
        raise ValueError(f"CSV must contain target column '{TARGET}' (0=not creditworthy, 1=creditworthy).")
    X, y = df.drop(columns=[TARGET]), df[TARGET].astype(int)
    if y.nunique() < 2:
        raise ValueError("Target must contain both classes 0 and 1.")
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    prep = ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical)
    ])
    models = {
        "logistic_regression": LogisticRegression(max_iter=1500, class_weight="balanced"),
        "decision_tree": DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                random_state=42)
    }
    if model_name not in models:
        raise ValueError(f"Choose one of: {', '.join(models)}")
    model = Pipeline([("preprocessing", prep), ("classifier", models[model_name])])
    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    print("\\nMODEL:", model_name)
    print("Accuracy:", round(accuracy_score(y_test, pred), 4))
    print("ROC-AUC:", round(roc_auc_score(y_test, proba), 4))
    print("\\nClassification report:\\n", classification_report(y_test, pred, zero_division=0))
    print("Confusion matrix:\\n", confusion_matrix(y_test, pred))
    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, "models/credit_scoring_model.joblib")
    print("\\nSaved model to models/credit_scoring_model.joblib")

def main():
    parser = argparse.ArgumentParser(description="CodeAlpha Task 1: Credit Scoring Model")
    parser.add_argument("--data", default="data/credit_data.csv", help="Path to input CSV")
    parser.add_argument("--model", default="random_forest",
                        choices=["logistic_regression", "decision_tree", "random_forest"])
    parser.add_argument("--make-demo-data", action="store_true")
    args = parser.parse_args()
    if args.make_demo_data or not Path(args.data).exists():
        make_demo_data(args.data)
    train(args.data, args.model)

if __name__ == "__main__":
    main()
