#!/usr/bin/env python
"""
Train a baseline classifier to predict Vaporix fault events.

The script expects the engineered feature CSV created by
`analysis/vaporix_exploration.py`. It splits the data, trains a
RandomForest model, prints evaluation metrics, and optionally writes
metrics/model artefacts to disk.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
except ImportError:  # pragma: no cover - optional dependency
    ImbPipeline = None
    SMOTE = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline Vaporix fault-event classifier")
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("reports/vaporix_features.csv"),
        help="Pfad zur gelabelten Feature-Datei (CSV)",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="fault_event",
        help="Zielspalte (default: fault_event)",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Testdatenanteil (default 0.2)")
    parser.add_argument("--random-state", type=int, default=42, help="Random-State für Reproduzierbarkeit")
    parser.add_argument(
        "--exclude-features",
        type=str,
        nargs="*",
        default=[
            "fault_event",
            "fault_active",
            "error_counter",
            "error_delta",
            "is_alert",
            "anomaly_score",
        ],
        help="Feature-Spalten, die nicht als Eingabe genutzt werden sollen (Standard: potentielle Leakage-Spalten).",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        help="Optionaler Pfad, um das trainierte Modell via joblib zu speichern",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        help="Optionaler Pfad, um Evaluationsmetriken als JSON zu speichern",
    )
    parser.add_argument(
        "--model-type",
        choices=["random_forest", "hist_gradient_boosting", "logistic"],
        default="random_forest",
        help="Welcher Klassifikator verwendet werden soll (Default: RandomForest).",
    )
    parser.add_argument(
        "--use-smote",
        action="store_true",
        help="Setzt SMOTE zur Klassenbalancierung auf dem Trainingssplit ein (benötigt imbalanced-learn).",
    )
    return parser.parse_args()


def _detect_feature_columns(
    df: pd.DataFrame, target: str, exclude: List[str]
) -> Tuple[List[str], List[str]]:
    """Trennt numerische von kategorialen Feature-Spalten."""
    excluded = {
        target,
        "alert_reasons",
        "alert_reason_str",
        "severity",
    }
    excluded.update(exclude)
    candidate_cols = [c for c in df.columns if c not in excluded]

    numeric_cols: List[str] = []
    categorical_cols: List[str] = []

    for col in candidate_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    return numeric_cols, categorical_cols


def train_model(args: argparse.Namespace) -> None:
    if not args.features.exists():
        raise FileNotFoundError(f"Feature-Datei nicht gefunden: {args.features}")

    df = pd.read_csv(args.features, parse_dates=["date", "created_at"], keep_default_na=True)
    if args.target not in df.columns:
        raise KeyError(f"Zielspalte '{args.target}' nicht gefunden.")

    df = df.dropna(subset=[args.target])
    df[args.target] = df[args.target].astype(int)

    numeric_cols, categorical_cols = _detect_feature_columns(df, args.target, args.exclude_features)

    # Bool-Spalten als numerisch behandeln
    for col in categorical_cols[:]:
        if set(df[col].dropna().unique()).issubset({0, 1, True, False}):
            df[col] = df[col].astype(float)
            numeric_cols.append(col)
            categorical_cols.remove(col)

    X = df[numeric_cols + categorical_cols]
    y = df[args.target]

    def _fallback_split() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        return train_test_split(
            X,
            y,
            test_size=args.test_size,
            stratify=y,
            random_state=args.random_state,
        )

    if "sensor_no" in df.columns:
        groups = df["sensor_no"].astype(str)
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=args.test_size, random_state=args.random_state
        )
        try:
            train_idx, test_idx = next(splitter.split(X, y, groups))
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        except ValueError:
            print("GroupShuffleSplit fehlgeschlagen – fallback auf klassisches Stratified-Split.")
            X_train, X_test, y_train, y_test = _fallback_split()

        if y_test.nunique() < 2:
            print(
                "Warnung: Testsplit enthält nur eine Klasse. "
                "Fallback auf klassisches Stratified-Split."
            )
            X_train, X_test, y_train, y_test = _fallback_split()
    else:
        X_train, X_test, y_train, y_test = _fallback_split()

    numeric_transformer = SkPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = SkPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    smote_step = None
    class_weight = "balanced"
    if args.use_smote:
        if SMOTE is None or ImbPipeline is None:
            raise ImportError("imbalanced-learn ist nicht installiert. Bitte `pip install imbalanced-learn` ausführen.")
        positive_count = int(y_train.sum())
        if positive_count < 2:
            print("Zu wenige positive Beispiele für SMOTE – überspringe Oversampling.")
        else:
            k_neighbors = min(5, max(1, positive_count - 1))
            smote_step = SMOTE(
                random_state=args.random_state,
                k_neighbors=k_neighbors,
            )
            class_weight = None  # Oversampling gleicht Klassen aus

    steps = [("preprocess", preprocessor)]
    if smote_step is not None:
        steps.append(("smote", smote_step))

    if args.model_type == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=400,
            random_state=args.random_state,
            class_weight=class_weight,
            n_jobs=-1,
            max_depth=None,
            min_samples_leaf=2,
        )
    elif args.model_type == "hist_gradient_boosting":
        clf = HistGradientBoostingClassifier(
            random_state=args.random_state,
            class_weight=class_weight,
            max_depth=10,
            learning_rate=0.1,
            max_iter=600,
        )
    elif args.model_type == "logistic":
        clf = LogisticRegression(
            max_iter=2000,
            class_weight=class_weight,
            solver="lbfgs",
        )
    else:  # pragma: no cover - fallback
        raise ValueError(f"Unbekannter model_type: {args.model_type}")

    steps.append(("clf", clf))

    pipeline_cls = ImbPipeline if smote_step is not None else SkPipeline
    model = pipeline_cls(steps=steps)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    report_dict = classification_report(y_test, y_pred, output_dict=True)
    roc_auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC: {roc_auc:.3f}")
    print("Confusion Matrix:")
    print(cm)

    feature_names = model.named_steps["preprocess"].get_feature_names_out()
    clf_estimator = model.named_steps["clf"]
    if hasattr(clf_estimator, "feature_importances_"):
        importances = clf_estimator.feature_importances_
        top_idx = np.argsort(importances)[::-1][:15]
        top_features = [
            {"feature": feature_names[i], "importance": float(importances[i])}
            for i in top_idx
        ]
    elif hasattr(clf_estimator, "coef_"):
        importances = np.abs(clf_estimator.coef_).ravel()
        top_idx = np.argsort(importances)[::-1][:15]
        top_features = [
            {"feature": feature_names[i], "importance": float(importances[i])}
            for i in top_idx
        ]
    else:
        importances = None
        top_features = []

    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    f1_scores = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )
    best_idx = int(np.argmax(f1_scores))
    best_threshold = float(thresholds[best_idx - 1]) if best_idx > 0 else 0.5
    best_f1 = float(f1_scores[best_idx])

    # Per-sensor metrics (falls Sensor-ID verfügbar)
    per_sensor = None
    if "sensor_no" in X_test.columns:
        eval_df = pd.DataFrame(
            {
                "sensor_no": X_test["sensor_no"],
                "y_true": y_test,
                "y_pred": y_pred,
                "y_prob": y_prob,
            }
        )
        eval_df["tp"] = (eval_df["y_true"] & eval_df["y_pred"]).astype(int)
        per_sensor_df = eval_df.groupby("sensor_no").agg(
            support=("y_true", "count"),
            positives=("y_true", "sum"),
            detected=("y_pred", "sum"),
            true_positives=("tp", "sum"),
        )
        per_sensor_df["recall"] = per_sensor_df.apply(
            lambda row: row["true_positives"] / row["positives"] if row["positives"] else np.nan,
            axis=1,
        )
        per_sensor_df["precision"] = per_sensor_df.apply(
            lambda row: row["true_positives"] / row["detected"] if row["detected"] else np.nan,
            axis=1,
        )
        per_sensor_df = per_sensor_df.replace({np.nan: None})
        per_sensor = per_sensor_df.reset_index().to_dict(orient="records")

    metrics = {
        "roc_auc": float(roc_auc),
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "top_features": top_features,
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "precision_recall_curve": {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "thresholds": thresholds.tolist(),
            "best_threshold": best_threshold,
            "best_f1": best_f1,
        },
        "per_sensor": per_sensor,
    }

    if args.metrics_out:
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        with args.metrics_out.open("w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2)
        print(f"Metriken gespeichert unter {args.metrics_out}")

    if args.model_out:
        args.model_out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, args.model_out)
        print(f"Modell gespeichert unter {args.model_out}")


if __name__ == "__main__":
    train_model(parse_args())
