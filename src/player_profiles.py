"""Build player style profiles from processed action-value data."""

from __future__ import annotations

from pathlib import Path
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


IDENTIFIER_COLUMNS = ["player_id", "player_name", "team_id", "team_name"]
PROFILE_FEATURE_COLUMNS = [
    "action_volume",
    "total_xt",
    "xt_per90",
    "xt_per_action",
    "pass_xt_total",
    "carry_dribble_xt_total",
    "pass_xt_share",
    "carry_dribble_xt_share",
    "progression_xt_share",
    "final_third_xt_share",
]


def _validate_dataframe(df: pd.DataFrame, required_columns: list[str], name: str) -> None:
    """Raise a clear error when an input table is missing key fields."""
    if df is None or df.empty:
        raise ValueError(f"`{name}` is empty. A populated DataFrame is required.")

    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"`{name}` is missing required columns: {', '.join(missing)}")


def _identifier_columns(df: pd.DataFrame) -> list[str]:
    """Return the identifier columns present in a DataFrame."""
    return [column for column in IDENTIFIER_COLUMNS if column in df.columns]


def _feature_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric profile columns, excluding identifiers and labels."""
    excluded = set(IDENTIFIER_COLUMNS + ["cluster"])
    numeric_columns = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    return [column for column in numeric_columns if column not in excluded]


def build_player_feature_matrix(actions_df: pd.DataFrame, player_summary_df: pd.DataFrame) -> pd.DataFrame:
    """Build a player feature table from action-level xT outputs.

    The goal is to describe how a player creates value: through passes,
    carries, overall volume, and the balance between progression and
    final-third threat.
    """
    xt_column = "xt_added" if "xt_added" in actions_df.columns else "xt_delta" if "xt_delta" in actions_df.columns else None
    if xt_column is None:
        raise ValueError("`actions_df` must contain either `xt_added` or `xt_delta`.")
    _validate_dataframe(
        actions_df,
        ["player_id", "player_name", "team_id", "team_name", "type_name", xt_column],
        "actions_df",
    )
    _validate_dataframe(
        player_summary_df,
        ["player_id", "player_name", "team_id", "team_name", "action_count", "total_xt"],
        "player_summary_df",
    )

    summary_df = player_summary_df.copy()
    if "xt_per90" not in summary_df.columns and "xt_per_90" in summary_df.columns:
        summary_df["xt_per90"] = summary_df["xt_per_90"]
    if "action_count" not in summary_df.columns and "actions" in summary_df.columns:
        summary_df["action_count"] = summary_df["actions"]

    features = summary_df.loc[
        :,
        [column for column in IDENTIFIER_COLUMNS if column in summary_df.columns]
        + [column for column in ["action_count", "total_xt", "xt_per90"] if column in summary_df.columns],
    ].copy()
    features = features.rename(columns={"action_count": "action_volume"})

    progressive_mask = (
        actions_df["is_progressive_action"].fillna(False)
        if "is_progressive_action" in actions_df.columns
        else pd.Series(False, index=actions_df.index)
    )
    final_third_mask = (
        actions_df["is_final_third_action"].fillna(False)
        if "is_final_third_action" in actions_df.columns
        else pd.Series(False, index=actions_df.index)
    )

    pass_xt = (
        actions_df.loc[actions_df["type_name"] == "pass"]
        .groupby(["player_id", "team_id"], dropna=False)[xt_column]
        .sum()
        .reset_index(name="pass_xt_total")
    )
    carry_dribble_xt = (
        actions_df.loc[actions_df["type_name"].isin(["dribble", "take_on"])]
        .groupby(["player_id", "team_id"], dropna=False)[xt_column]
        .sum()
        .reset_index(name="carry_dribble_xt_total")
    )
    progression_xt = (
        actions_df.loc[progressive_mask]
        .groupby(["player_id", "team_id"], dropna=False)[xt_column]
        .sum()
        .reset_index(name="progression_xt_total")
    )
    final_third_xt = (
        actions_df.loc[final_third_mask]
        .groupby(["player_id", "team_id"], dropna=False)[xt_column]
        .sum()
        .reset_index(name="final_third_xt_total")
    )

    for addon in [pass_xt, carry_dribble_xt, progression_xt, final_third_xt]:
        features = features.merge(addon, on=["player_id", "team_id"], how="left")

    for column in [
        "pass_xt_total",
        "carry_dribble_xt_total",
        "progression_xt_total",
        "final_third_xt_total",
    ]:
        features[column] = features[column].fillna(0.0)

    features["xt_per_action"] = features["total_xt"] / features["action_volume"].clip(lower=1)
    xt_total_safe = features["total_xt"].replace(0, np.nan)
    features["pass_xt_share"] = features["pass_xt_total"] / xt_total_safe
    features["carry_dribble_xt_share"] = features["carry_dribble_xt_total"] / xt_total_safe
    features["progression_xt_share"] = features["progression_xt_total"] / xt_total_safe
    features["final_third_xt_share"] = features["final_third_xt_total"] / xt_total_safe

    share_columns = [
        "pass_xt_share",
        "carry_dribble_xt_share",
        "progression_xt_share",
        "final_third_xt_share",
    ]
    features[share_columns] = features[share_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    ordered_columns = _identifier_columns(features) + [
        column for column in PROFILE_FEATURE_COLUMNS if column in features.columns
    ]
    return features.loc[:, ordered_columns].sort_values("total_xt", ascending=False).reset_index(drop=True)


def normalize_features(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Standardize player profile features so they can be compared fairly."""
    if feature_df is None or feature_df.empty:
        return pd.DataFrame()

    feature_columns = _feature_columns(feature_df)
    if not feature_columns:
        raise ValueError("`feature_df` does not contain numeric profile features to normalize.")

    identifiers = feature_df.loc[:, _identifier_columns(feature_df)].copy()
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(feature_df[feature_columns].fillna(0.0))
    scaled_df = pd.DataFrame(scaled_values, columns=feature_columns, index=feature_df.index)
    return pd.concat([identifiers, scaled_df], axis=1)


def cluster_players(feature_df: pd.DataFrame, n_clusters: int = 6) -> pd.DataFrame:
    """Cluster players into broad style groups from the feature matrix."""
    if feature_df is None or feature_df.empty:
        return pd.DataFrame()
    if n_clusters < 2:
        raise ValueError("`n_clusters` must be at least 2.")

    normalized_df = normalize_features(feature_df)
    feature_columns = _feature_columns(normalized_df)
    if len(normalized_df) < n_clusters:
        raise ValueError("`n_clusters` cannot exceed the number of players in the feature matrix.")

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    clustered = feature_df.copy()
    clustered["cluster"] = model.fit_predict(normalized_df[feature_columns])
    sort_column = "total_xt" if "total_xt" in clustered.columns else "xt_total"
    return clustered.sort_values(["cluster", sort_column], ascending=[True, False]).reset_index(drop=True)


def find_similar_players(feature_df: pd.DataFrame, player_name: str, top_n: int = 10) -> pd.DataFrame:
    """Find players with the most similar profile to a chosen player."""
    if feature_df is None or feature_df.empty:
        return pd.DataFrame()
    if not player_name:
        raise ValueError("`player_name` must be provided.")
    if top_n <= 0:
        raise ValueError("`top_n` must be greater than 0.")
    if "player_name" not in feature_df.columns:
        raise ValueError("`feature_df` must contain a `player_name` column.")

    normalized_df = normalize_features(feature_df)
    feature_columns = _feature_columns(normalized_df)

    matches = normalized_df.loc[
        normalized_df["player_name"].str.casefold() == player_name.casefold()
    ]
    if matches.empty:
        raise ValueError(f"Player '{player_name}' was not found in the feature matrix.")

    player_index = matches.index[0]
    neighbors = NearestNeighbors(metric="euclidean")
    neighbors.fit(normalized_df[feature_columns])

    n_neighbors = min(len(normalized_df), top_n + 1)
    distances, indices = neighbors.kneighbors(
        normalized_df.loc[[player_index], feature_columns],
        n_neighbors=n_neighbors,
    )

    similar = feature_df.iloc[indices[0]].copy()
    similar["distance"] = distances[0]
    similar = similar.loc[similar["player_name"].str.casefold() != player_name.casefold()]
    return similar.sort_values("distance").head(top_n).reset_index(drop=True)


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    actions_path = repo_root / "data" / "processed" / "actions.parquet"
    players_path = repo_root / "data" / "processed" / "player_summary.parquet"

    actions_df = pd.read_parquet(actions_path)
    player_summary_df = pd.read_parquet(players_path)

    feature_df = build_player_feature_matrix(actions_df, player_summary_df)
    clustered_df = cluster_players(feature_df, n_clusters=6)
    sample_player = feature_df.iloc[0]["player_name"]
    similar_players_df = find_similar_players(feature_df, sample_player, top_n=5)

    print("Player feature matrix:")
    print(feature_df.head(5).to_string(index=False))
    print("\nClustered players:")
    print(clustered_df[["player_name", "team_name", "cluster", "total_xt"]].head(10).to_string(index=False))
    print(f"\nPlayers most similar to {sample_player}:")
    print(similar_players_df[["player_name", "team_name", "distance"]].to_string(index=False))
