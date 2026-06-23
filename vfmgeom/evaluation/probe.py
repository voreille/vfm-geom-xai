from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProbeResult:
    balanced_accuracy: float
    accuracy: float
    chance_balanced_accuracy: float
    n_train: int
    n_test: int
    n_classes: int
    classes: list[str]


def make_probe_classifier(
    probe_type: str = "logistic",
) -> Pipeline:
    if probe_type == "logistic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
            ),
        )
    elif probe_type == "sgd":
        return make_pipeline(
            StandardScaler(),
            SGDClassifier(
                loss="log_loss",
                penalty="l2",
                alpha=1e-4,
                class_weight="balanced",
                max_iter=500,
                tol=1e-3,
                random_state=0,
                n_jobs=-1,
            ),
        )
    else:
        raise ValueError(f"Unsupported probe type: {probe_type}")


def evaluate_probe_train_test(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray | pd.Series,
    y_test: np.ndarray | pd.Series,
    probe_type: str = "logistic",
) -> ProbeResult:
    y_train = pd.Series(y_train).astype(str).to_numpy()
    y_test = pd.Series(y_test).astype(str).to_numpy()

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)

    unknown = sorted(set(y_test) - set(label_encoder.classes_))
    if unknown:
        raise ValueError(
            f"Test contains labels not present in train: {unknown}"
        )

    y_test = label_encoder.transform(y_test)

    clf = make_probe_classifier(
        probe_type=probe_type,
    )
    clf.fit(x_train, y_train)

    if probe_type == "logistic":
        logistic = clf.named_steps["logisticregression"]
        logger.info(
            "Logistic regression iterations: %s",
            logistic.n_iter_,
        )
    elif probe_type == "sgd":
        sgd = clf.named_steps["sgdclassifier"]
        logger.info(
            "SGD classifier iterations: %s",
            sgd.n_iter_,
        )

    y_pred = clf.predict(x_test)

    return ProbeResult(
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
