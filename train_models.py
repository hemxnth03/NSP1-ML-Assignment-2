"""Train, tune, evaluate and save the five classifiers.

Run once (on the BITS Virtual Lab):   python train_models.py

Outputs
  model/*.joblib          one fitted sklearn Pipeline per classifier + meta.joblib
  test_data.csv           20 % stratified hold-out (features + Target) for the app
  results/metrics.csv     the comparison table used in README.md
  results/best_params.json
  requirements.txt        pinned to the library versions used for training
"""
import json
import sys
from importlib import metadata
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from metrics_utils import METRIC_ORDER, evaluate_model

import os
import warnings

os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"   # inherited by the parallel grid-search workers
warnings.filterwarnings("ignore", category=FutureWarning)

# ----------------------------------------------------------------------------
# Settings -- fixed random state for reproducibility.
# Keep this value unchanged when reproducing the reported experiment.
# ----------------------------------------------------------------------------
RANDOM_STATE = 2607
TEST_SIZE = 0.20          # the dataset authors also used an 80/20 split
LOCAL_CSV = Path("data.csv")   # Required for local model training/retraining
TARGET = "Target"

# Integer-CODED categories (a code like Course=9003 is a label, not a quantity).
# These get one-hot encoding for the linear / distance / margin models.
NOMINAL_COLUMNS = [
    "Marital status", "Application mode", "Course", "Previous qualification",
    "Nacionality", "Mother's qualification", "Father's qualification",
    "Mother's occupation", "Father's occupation",
]

MODEL_DIR = Path("model")
RESULT_DIR = Path("results")
MODEL_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)


def load_data() -> pd.DataFrame:
    """Load UCI's data.csv (semicolon separated) from the project folder."""
    if not LOCAL_CSV.exists():
        sys.exit("data.csv not found next to train_models.py.")
    df = pd.read_csv(LOCAL_CSV, sep=";")
    if df.shape[1] == 1:               # not semicolon separated
        df = pd.read_csv(LOCAL_CSV, sep=",")
    print(f"Loaded {LOCAL_CSV}")
    df.columns = df.columns.str.strip()    # UCI ships a column with a trailing tab
    return df


def build_search_spaces(nominal, binary, numeric):
    """(estimator, parameter grid) for each required model.

    Preprocessing lives INSIDE the saved Pipelines, so the app can feed raw rows.
      * Logistic Regression, KNN, SVM: one-hot encoding for nominal categorical
        features, binary columns passed through, and StandardScaler for numeric
        and ordinal features.
      * Decision Tree: uses raw columns because feature scaling is not required
        for tree-based splits. Integer-coded nominal features are still treated
        as ordered numerical values, which is a limitation.
      * Gaussian Naive Bayes: uses raw columns as a simplified baseline.
        One-hot encoded 0/1 indicators are discrete rather than Gaussian.
        However, raw integer-coded nominal values are also treated as numeric,
        which is another limitation of this approach.
    """
    def encoded(clf):
        prep = ColumnTransformer([
            ("nominal", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                      min_frequency=20, sparse_output=False), nominal),
            ("binary", "passthrough", binary),
            ("numeric", StandardScaler(), numeric),
        ])
        return Pipeline([("prep", prep), ("clf", clf)])

    return {
        "Logistic Regression": (
            encoded(LogisticRegression(max_iter=3000)),
            {"clf__C": [0.05, 0.3, 1, 5]},
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=RANDOM_STATE),
            {"max_depth": [3, 5, 7, 10], "min_samples_leaf": [5, 15, 30]},
        ),
        "KNN": (
            encoded(KNeighborsClassifier()),
            {"clf__n_neighbors": [7, 11, 17, 25], "clf__weights": ["uniform", "distance"]},
        ),
        "Naive Bayes": (
            GaussianNB(),
            {"var_smoothing": [1e-9, 1e-6, 1e-4, 1e-2]},
        ),
        "Support Vector Machine": (
            encoded(SVC(probability=True, random_state=RANDOM_STATE)),
            {"clf__C": [0.5, 2, 8], "clf__gamma": ["scale", 0.01]},
        ),
    }


def file_name(model_name: str) -> str:
    return model_name.lower().replace(" ", "_") + ".joblib"


def write_requirements():
    pkgs = ["streamlit", "scikit-learn", "numpy", "pandas", "matplotlib",
            "seaborn", "joblib"]
    lines = [f"{p}=={metadata.version(p)}" for p in pkgs]
    Path("requirements.txt").write_text("\n".join(lines) + "\n")
    print("Wrote requirements.txt (pinned):\n  " + "\n  ".join(lines))


def main():
    df = load_data()
    n_features = df.shape[1] - 1
    print(f"Rows: {len(df)}, features: {n_features}, classes: {df[TARGET].nunique()}")
    assert n_features >= 12, "Assignment needs >= 12 features"
    assert len(df) >= 500, "Assignment needs >= 500 instances"
    assert not df.isna().any().any(), "Unexpected missing values"

    encoder = LabelEncoder().fit(df[TARGET])
    class_names = list(encoder.classes_)
    n_classes = len(class_names)
    X = df.drop(columns=[TARGET])
    missing = [c for c in NOMINAL_COLUMNS if c not in X.columns]
    assert not missing, f"Nominal columns not found: {missing}"
    binary_cols = [c for c in X.columns if c not in NOMINAL_COLUMNS and X[c].nunique() == 2]
    numeric_cols = [c for c in X.columns if c not in NOMINAL_COLUMNS + binary_cols]
    print(f"Column groups -> nominal: {len(NOMINAL_COLUMNS)}, binary: {len(binary_cols)}, "
          f"numeric/ordinal: {len(numeric_cols)}")
    y = pd.Series(encoder.transform(df[TARGET]), index=df.index)
    print("Class balance:\n", df[TARGET].value_counts().to_string())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    # Hold-out file for the Streamlit app: raw features + the text label.
    test_out = X_test.copy()
    test_out[TARGET] = encoder.inverse_transform(y_test)
    test_out.to_csv("test_data.csv", index=False)
    print(f"Saved test_data.csv ({len(test_out)} rows)")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows, best_params = [], {}
    for name, (estimator, grid) in build_search_spaces(NOMINAL_COLUMNS, binary_cols, numeric_cols).items():
        # Tuning uses the TRAINING split only; the hold-out is never seen here.
        search = GridSearchCV(estimator, grid, cv=cv, scoring="f1_macro", n_jobs=-1)
        search.fit(X_train, y_train)
        model = search.best_estimator_
        joblib.dump(model, MODEL_DIR / file_name(name))

        metrics, _ = evaluate_model(model, X_test, y_test, n_classes)
        rows.append({"ML Model Name": name, **metrics})
        best_params[name] = {"params": search.best_params_, "cv_f1_macro": search.best_score_}
        print(f"{name:24s} " + "  ".join(f"{k}={metrics[k]:.4f}" for k in METRIC_ORDER))

    joblib.dump(
        {
            "feature_names": list(X.columns),
            "class_names": class_names,
            "target": TARGET,
            "sklearn_version": metadata.version("scikit-learn"),
            "models": {n: file_name(n) for n in best_params},
        },
        MODEL_DIR / "meta.joblib",
    )

    table = pd.DataFrame(rows)
    table.to_csv(RESULT_DIR / "metrics.csv", index=False)
    (RESULT_DIR / "best_params.json").write_text(json.dumps(best_params, indent=2, default=str))
    print("\nMarkdown table for README.md:\n")
    t = table.round(4)
    print("| " + " | ".join(t.columns) + " |")
    print("|" + "---|" * len(t.columns))
    for _, r in t.iterrows():
        print("| " + " | ".join(str(v) for v in r) + " |")
    write_requirements()


if __name__ == "__main__":
    sys.exit(main())