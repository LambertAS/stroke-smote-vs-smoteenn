"""Encoding maps, BMI imputation, train/test split and scaling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

TARGET = "stroke"

# Fixed label encodings, in the order the original notebook found the categories.
ENCODING_MAPS: dict[str, dict[str, int]] = {
    "gender": {"Male": 0, "Female": 1, "Other": 2},
    "ever_married": {"Yes": 0, "No": 1},
    "work_type": {
        "Private": 0,
        "Self-employed": 1,
        "Govt_job": 2,
        "children": 3,
        "Never_worked": 4,
    },
    "smoking_status": {"formerly smoked": 0, "never smoked": 1, "smokes": 2, "Unknown": 3},
    "Residence_type": {"Urban": 0, "Rural": 1},
}


@dataclass
class HoldoutData:
    """Scaled train/test arrays plus the unscaled frames they came from."""

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    X_train_raw: pd.DataFrame
    X_test_raw: pd.DataFrame
    feature_names: list[str]
    scaler: MinMaxScaler


def encode(df: pd.DataFrame) -> pd.DataFrame:
    """Apply ``ENCODING_MAPS``; raise if any category is not covered."""
    out = df.copy()
    for col, mapping in ENCODING_MAPS.items():
        out[col] = out[col].map(mapping)
        if out[col].isna().any():
            raise ValueError(f"Unmapped categories in {col}")
        out[col] = out[col].astype("int64")
    return out


def impute_bmi(frame: pd.DataFrame, mean: float) -> pd.DataFrame:
    """Fill missing BMI with a given mean."""
    out = frame.copy()
    out["bmi"] = out["bmi"].fillna(mean)
    return out


def prepare_holdout(raw: pd.DataFrame, cfg: dict) -> HoldoutData:
    """Run the preprocessing of the chosen profile and return scaled train/test arrays."""
    df = raw.drop(columns="id")
    strategy = cfg["preprocessing"]["bmi_imputation"]
    if strategy == "full_data_mean":
        df = impute_bmi(df, df["bmi"].mean())
    df = encode(df)

    X, y = df.drop(columns=TARGET), df[TARGET]
    split = cfg["split"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=split["test_size"],
        random_state=split["random_state"],
        stratify=y if split["stratify"] else None,
    )
    if strategy == "train_mean":
        imputer = SimpleImputer(strategy="mean").fit(X_train[["bmi"]])
        train_mean = float(imputer.statistics_[0])
        X_train, X_test = impute_bmi(X_train, train_mean), impute_bmi(X_test, train_mean)
    elif strategy != "full_data_mean":
        raise ValueError(f"Unknown bmi_imputation: {strategy}")

    scaler = MinMaxScaler()
    return HoldoutData(
        X_train=scaler.fit_transform(X_train),
        X_test=scaler.transform(X_test),
        y_train=y_train.to_numpy(),
        y_test=y_test.to_numpy(),
        X_train_raw=X_train,
        X_test_raw=X_test,
        feature_names=list(X.columns),
        scaler=scaler,
    )
