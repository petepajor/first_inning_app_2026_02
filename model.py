"""
model.py — trains and caches the GBM model from historical CSVs/XLSXs.
Call train_model(pit_df) to get back a fitted pipeline.
Call predict(pipeline, pit_df) to score new pitchers.
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

PITCHER_FEATURE_COLS = [
    "K%", "BB%", "K-BB%", "wOBA", "ERA", "FIP", "xFIP",
    "BABIP", "GB%", "LD%", "FB%", "HR/FB", "WHIP"
]

ROLLING_WINDOW = 10
MODEL_CACHE = "model_cache.pkl"


def _to_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _build_rolling_features(df):
    """Add rolling 10-start pitcher features, shifted to avoid leakage."""
    df = df.copy()
    df = _to_numeric(df, PITCHER_FEATURE_COLS + ["R"])
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Name", "Date"])
    df["allowed_run"] = (df["R"] > 0).astype(int)

    for c in PITCHER_FEATURE_COLS:
        if c in df.columns:
            df[f"roll_{c}"] = df.groupby("Name")[c].transform(
                lambda x: x.shift(1).rolling(ROLLING_WINDOW, min_periods=3).mean()
            )

    df["roll_run_rate"] = df.groupby("Name")["allowed_run"].transform(
        lambda x: x.shift(1).expanding(min_periods=3).mean()
    )
    return df


def get_feature_cols():
    return ["roll_run_rate"] + [f"roll_{c}" for c in PITCHER_FEATURE_COLS]


def train_model(pit_df: pd.DataFrame):
    """Train model from historical pitching data. Returns fitted pipeline."""
    df = _build_rolling_features(pit_df)
    feat_cols = get_feature_cols()
    df_model = df.dropna(subset=feat_cols + ["allowed_run"])

    X = df_model[feat_cols].values
    y = df_model["allowed_run"].values

    gbm = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42
    )
    gbm.fit(X, y)

    meta = {
        "model": gbm,
        "feature_cols": feat_cols,
        "baseline": float(y.mean()),
        "n_starts": int(len(df_model)),
        "n_pitchers": int(df_model["Name"].nunique()),
        "feature_importances": dict(
            zip(feat_cols, [round(float(v), 4) for v in gbm.feature_importances_])
        ),
    }
    return meta


def save_model(meta, path=MODEL_CACHE):
    with open(path, "wb") as f:
        pickle.dump(meta, f)


def load_model(path=MODEL_CACHE):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def get_pitcher_rolling_stats(pit_df: pd.DataFrame):
    """
    Given the historical pitching DF, return the most recent rolling
    feature snapshot per pitcher — used to score today's starters.
    """
    df = _build_rolling_features(pit_df)
    feat_cols = get_feature_cols()
    latest = (
        df.sort_values("Date")
        .groupby("Name")
        .last()
        .reset_index()[["Name", "Tm", "Date"] + feat_cols + ["allowed_run"]]
    )
    return latest


def score_pitchers(meta, pitcher_names: list, pit_df: pd.DataFrame):
    """
    Score a list of pitcher names against the trained model.
    Returns a DataFrame with Name, prob, implied odds, etc.
    """
    rolling = get_pitcher_rolling_stats(pit_df)
    feat_cols = meta["feature_cols"]

    results = []
    for name in pitcher_names:
        row = rolling[rolling["Name"].str.lower() == name.lower()]
        if row.empty:
            results.append({"Name": name, "found": False})
            continue
        row = row.iloc[0]
        feats = row[feat_cols].values.astype(float)
        if np.isnan(feats).any():
            results.append({"Name": name, "found": False, "reason": "insufficient history"})
            continue
        prob = float(meta["model"].predict_proba([feats])[0, 1])
        results.append({
            "Name": name,
            "Team": row.get("Tm", "—"),
            "found": True,
            "prob": prob,
            "actual_rate": float(row.get("allowed_run", np.nan)),
            **{c: row[c] for c in feat_cols},
        })

    return pd.DataFrame(results)


def prob_to_american(p: float) -> str:
    """Convert probability to American odds string."""
    if p <= 0 or p >= 1:
        return "—"
    if p >= 0.5:
        return f"-{round((p / (1 - p)) * 100)}"
    else:
        return f"+{round(((1 - p) / p) * 100)}"


def american_to_implied(odds_str: str) -> float | None:
    """Convert American odds string to implied probability."""
    try:
        odds = int(odds_str.replace("+", "").replace(" ", ""))
        if odds > 0:
            return 100 / (odds + 100)
        else:
            return abs(odds) / (abs(odds) + 100)
    except Exception:
        return None


def edge(model_prob: float, book_implied: float) -> float:
    """Edge = model probability minus book implied probability."""
    return model_prob - book_implied
