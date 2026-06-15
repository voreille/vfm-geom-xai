# vfmgeom/evaluation/scanner_probe.py

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


@dataclass(frozen=True)
class ScannerProbeResult:
    balanced_accuracy: float
    accuracy: float
    chance_balanced_accuracy: float
    n_train: int
    n_test: int
    n_classes: int
    classes: list[str]


def make_scanner_probe_classifier():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
        ),
    )


def evaluate_scanner_probe_train_test(
    x_train: np.ndarray,
    x_test: np.ndarray,
    scanner_train: np.ndarray | pd.Series,
    scanner_test: np.ndarray | pd.Series,
) -> ScannerProbeResult:
    scanner_train = pd.Series(scanner_train).astype(str).to_numpy()
    scanner_test = pd.Series(scanner_test).astype(str).to_numpy()

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(scanner_train)

    unknown = sorted(set(scanner_test) - set(label_encoder.classes_))
    if unknown:
        raise ValueError(
            f"Test contains scanner labels not present in train: {unknown}"
        )

    y_test = label_encoder.transform(scanner_test)

    clf = make_scanner_probe_classifier()
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)

    return ScannerProbeResult(
        balanced_accuracy=float(balanced_accuracy_score(y_test, y_pred)),
        accuracy=float(accuracy_score(y_test, y_pred)),
        chance_balanced_accuracy=float(1.0 / len(label_encoder.classes_)),
        n_train=int(len(x_train)),
        n_test=int(len(x_test)),
        n_classes=int(len(label_encoder.classes_)),
        classes=label_encoder.classes_.tolist(),
    )


def summarize_probe_by_rank(
    fold_scores: pd.DataFrame,
    projected_col: str,
    raw_col: str = "raw_score",
) -> pd.DataFrame:
    df = fold_scores.copy()
    df["score_delta"] = df[projected_col] - df[raw_col]

    return (
        df.groupby("rank")
        .agg(
            raw_score_mean=(raw_col, "mean"),
            raw_score_std=(raw_col, "std"),
            projected_score_mean=(projected_col, "mean"),
            projected_score_std=(projected_col, "std"),
            score_delta_mean=("score_delta", "mean"),
            score_delta_std=("score_delta", "std"),
        )
        .reset_index()
    )
